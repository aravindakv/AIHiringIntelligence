"""Streamlit dashboard: run the pipeline, review insights (human in the loop), compare inputs, see cost/latency.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hiring_intel import hitl  # noqa: E402
from hiring_intel.contracts import AgentOutput  # noqa: E402
from hiring_intel.observability import query  # noqa: E402

st.set_page_config(page_title="Hiring Intelligence", layout="wide")
SEV_ORDER = {"high": 3, "medium": 2, "low": 1}
RAW = ROOT / "data/raw"
DATASETS = sorted(p.name for p in RAW.iterdir() if p.is_dir()) if RAW.exists() else []
if not DATASETS:
    st.error("No datasets in data/raw. Run: python scripts/generate_mock_data.py --all")
    st.stop()

# ------------------------------------------------------------------ sidebar: run controls
st.sidebar.header("Run the pipeline")
ds = st.sidebar.selectbox("Input dataset", DATASETS, index=DATASETS.index("baseline_medium")
                          if "baseline_medium" in DATASETS else 0)
variant = st.sidebar.selectbox("Variant", ["optimized", "routed", "routed_compressed", "all_large"])
inject = st.sidebar.selectbox("Fault injection (demo)", ["none", "fabricate numbers in sourcing", "crash offers agent"])
if st.sidebar.button("Run now", type="primary"):
    from hiring_intel.graph import run
    from scripts.benchmark import VARIANTS
    kw = {k: v for k, v in VARIANTS[variant].items() if k != "repeat"}
    scenario, size = ds.rsplit("_", 1)
    with st.spinner("Running 8 agents..."):
        s = run(f"data/raw/{ds}", scenario=scenario, size=size, variant=variant, hitl_mode="queue",
                inject_fault="sourcing" if inject.startswith("fabricate") else None,
                crash_agent="offers" if inject.startswith("crash") else None, **kw)
    st.session_state["run_id"] = s["run_id"]
    st.rerun()

runs = query("SELECT * FROM runs ORDER BY ts DESC")
if runs.empty:
    st.info("No runs yet. Pick a dataset and press **Run now**.")
    st.stop()
default = st.session_state.get("run_id", runs.run_id.iloc[0])
run_id = st.sidebar.selectbox("Show run", runs.run_id.tolist(),
                              index=runs.run_id.tolist().index(default) if default in runs.run_id.tolist() else 0)
meta = runs[runs.run_id == run_id].iloc[0]

outs = [AgentOutput.model_validate_json(j) for j in
        query("SELECT json FROM agent_outputs WHERE run_id=?", (run_id,)).json]
evals = {r.agent: json.loads(r.json) for r in query("SELECT agent, json FROM eval_results WHERE run_id=?",
                                                    (run_id,)).itertuples()}
calls = query("SELECT * FROM llm_calls WHERE run_id=?", (run_id,))
routes = query("SELECT * FROM routing WHERE run_id=?", (run_id,))
decisions = query("SELECT * FROM decisions WHERE run_id=?", (run_id,))
reviews = query("SELECT * FROM review_queue WHERE run_id=?", (run_id,))
status_of = {r.item_id: r.status for r in reviews.itertuples()}

st.title("Hiring Intelligence")
st.caption(f"Run `{run_id}` · input `{meta.data_path}` · variant `{meta.variant}` · "
           f"{int(meta.n_candidates):,} candidates")

tabs = st.tabs(["Overview", "Insights", "Human review", "Candidates", "Evaluation", "Cost & latency",
                "Compare inputs", "Run log"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Run cost", f"${meta.total_usd:.4f}")
    c[1].metric("End-to-end latency", f"{meta.e2e_latency_ms / 1000:.1f} s")
    c[2].metric("Mean eval score", f"{meta.mean_eval:.2f}")
    c[3].metric("Awaiting review", int((reviews.status == "pending").sum()))
    c[4].metric("Agents OK", f"{sum(o.status != 'failed' for o in outs)}/{len(outs)}")
    from hiring_intel.ingest import load_all
    from hiring_intel.metrics import STAGES
    d = load_all(meta.data_path)
    funnel = d.stage_events.groupby("stage").candidate_id.nunique().reindex(STAGES).reset_index()
    left, right = st.columns([1, 1])
    left.plotly_chart(px.funnel(funnel, x="candidate_id", y="stage", title="Hiring funnel"), width="stretch")
    top = sorted([(o.agent, k, i) for o in outs for k, i in enumerate(o.insights)],
                 key=lambda t: (SEV_ORDER[t[2].severity], t[2].confidence_score), reverse=True)[:4]
    with right:
        st.subheader("Top actions")
        for agent, k, i in top:
            badge = status_of.get(f"{run_id}:{agent}:{k}", "")
            st.info(f"**{i.title}** — {agent}, confidence {i.confidence_score:.0%}, *{badge}*\n\n{i.recommendation}")

# ------------------------------------------------------------------ insights
with tabs[1]:
    for o in outs:
        st.subheader(o.agent.replace("_", " ").title())
        if o.status == "failed":
            st.error(o.error)
            continue
        if o.status == "degraded":
            st.warning("No RAG context retrieved; confidence capped")
        for k, i in enumerate(o.insights):
            stt = status_of.get(f"{run_id}:{o.agent}:{k}", "")
            with st.expander(f"[{i.severity}] {i.title} — {i.confidence_score:.0%} · {stt}", expanded=k == 0):
                st.write(i.recommendation)
                st.progress(i.confidence_score)
                st.dataframe(pd.DataFrame([e.model_dump() for e in i.evidence]).astype({"value": str}), hide_index=True)
                if i.rebalance_plan:
                    st.dataframe(pd.DataFrame([m.model_dump() for m in i.rebalance_plan]), hide_index=True)
                st.caption(f"Alternative: {i.alternative.option} — {i.alternative.trade_off}")
        if o.cost_of_insight:
            k_ = o.cost_of_insight
            st.caption(f"{k_.model} ({k_.tier}) · {k_.input_tokens}+{k_.output_tokens} tokens · ${k_.usd:.5f} · "
                       f"{k_.latency_ms} ms{' (simulated)' if k_.simulated else ''} · compute {o.compute_ms} ms")

# ------------------------------------------------------------------ human review
with tabs[2]:
    st.write("Insights that are high-severity, low-confidence, flagged by the evaluator, or that OpenJev marks as "
             "changing comp/headcount/workload wait here. Approved ones are added to the RAG store.")
    reviewer = st.text_input("Reviewer name", value="hiring-manager")
    for it in hitl.pending(run_id):
        with st.container(border=True):
            st.markdown(f"**{it.insight.title}** · `{it.agent}` · confidence {it.insight.confidence_score:.0%}")
            st.caption("Why review: " + "; ".join(it.reasons))
            text = st.text_area("Recommendation", it.insight.recommendation, key=f"t-{it.item_id}")
            note = st.text_input("Note", key=f"n-{it.item_id}")
            a, b, c_ = st.columns(3)
            if a.button("Approve", key=f"a-{it.item_id}"):
                hitl.apply_decision(it.item_id, "edited" if text != it.insight.recommendation else "approved",
                                    reviewer, note, text)
                st.rerun()
            if b.button("Reject", key=f"r-{it.item_id}"):
                hitl.apply_decision(it.item_id, "rejected", reviewer, note)
                st.rerun()
    done = reviews[reviews.status != "pending"][["agent", "status", "reviewer", "note", "reasons"]]
    st.dataframe(done, hide_index=True)

# ------------------------------------------------------------------ candidates (demo lookup)
with tabs[3]:
    st.write("Look up a candidate by name or ID to follow their journey. Names are for people using this "
             "dashboard only: agents, prompts, LangSmith traces and DeepEval uploads never see them.")
    from hiring_intel.ingest import load_all as _load
    dd = _load(meta.data_path)

    def _dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        return df
    cand = dd.candidates.merge(dd.requisitions[["req_id", "role", "level", "hiring_manager"]]
                               if "hiring_manager" in dd.requisitions else dd.requisitions[["req_id", "role", "level"]],
                               on="req_id")
    if "candidate_name" not in cand:
        st.info("This dataset has no candidate names. Regenerate it with scripts/generate_mock_data.py.")
    else:
        q = st.text_input("Search name or ID", key="cand-q", placeholder="e.g. Isha Joshi or C0000001")
        c1, c2, c3 = st.columns(3)
        f_role = c1.multiselect("Role", sorted(cand.role.unique()), key="cand-role")
        f_out = c2.multiselect("Outcome", sorted(cand.outcome.unique()), key="cand-out")
        f_src = c3.multiselect("Source", sorted(cand.source.unique()), key="cand-src")
        view = cand
        if q:
            view = view[view.candidate_name.str.contains(q, case=False, regex=False)
                        | view.candidate_id.str.contains(q, case=False, regex=False)]
        for col, vals in (("role", f_role), ("outcome", f_out), ("source", f_src)):
            if vals:
                view = view[view[col].isin(vals)]
        dup = view.candidate_name.duplicated(keep=False).sum()
        st.caption(f"{len(view):,} candidates" + (f" · {dup} share a name with someone else in this list, "
                                                   "so pick by ID" if dup else ""))
        cols = ["candidate_id", "candidate_name", "role", "level", "source", "current_stage", "outcome"]
        st.dataframe(view[cols + (["hiring_manager"] if "hiring_manager" in view else [])].head(500),
                     hide_index=True)
        if len(view):
            pick = st.selectbox("Open candidate", view.candidate_id.head(500),
                                format_func=lambda c: f"{c} · {view.set_index('candidate_id').candidate_name[c]}",
                                key="cand-pick")
            row = cand.set_index("candidate_id").loc[pick]
            st.subheader(f"{row.candidate_name} ({pick})")
            st.write(f"{row.role}, {row.level} · source {row.source} · outcome **{row.outcome}**"
                     + (f" · hiring manager {row.hiring_manager}" if "hiring_manager" in row else ""))
            ev = dd.stage_events[dd.stage_events.candidate_id == pick]
            st.markdown("**Journey**")
            st.dataframe(_dates(ev[[c for c in ["stage", "entered_at", "exited_at", "result", "rejection_reason",
                                                "reason", "rejection_note"] if c in ev]]), hide_index=True)
            iv = dd.interviews[dd.interviews.candidate_id == pick]
            if len(iv):
                st.markdown("**Interviews**")
                st.dataframe(_dates(iv[[c for c in ["stage", "interviewer_id", "interviewer_name", "scheduled_at",
                                                    "score"] if c in iv]]), hide_index=True)
            of = dd.offers[dd.offers.candidate_id == pick]
            if len(of):
                st.markdown("**Offer**")
                o = of.iloc[0]
                st.write(f"Offered {o.offered_ctc:,.0f} vs expected {o.expected_ctc:,.0f} "
                         f"({o.offered_ctc / o.expected_ctc:.0%}) · **{o.status}**"
                         + (f" · {o.decline_note}" if isinstance(o.get("decline_note"), str) else ""))

# ------------------------------------------------------------------ evaluation
with tabs[4]:
    if evals:
        ev = pd.DataFrame([{"agent": a, "overall": e["overall"], "grounded": e["numbers_grounded"],
                            "refs_valid": e["refs_valid"], "flags": ", ".join(e["flags"]),
                            "judge": (e.get("judge") or {}).get("judge")} for a, e in evals.items()])
        st.plotly_chart(px.bar(ev, x="agent", y="overall", range_y=[0, 1], title="Eval score per agent"),
                        width="stretch")
        st.dataframe(ev, hide_index=True)
    for f in sorted((ROOT / "data/eval").glob("golden_*.csv")):
        g = pd.read_csv(f)
        st.write(f"**{f.stem}** — pass rate {g['pass'].mean():.0%}")
        st.dataframe(g, hide_index=True)

# ------------------------------------------------------------------ cost & latency
with tabs[5]:
    from hiring_intel.report import e2e_latency_ms
    lat = e2e_latency_ms(run_id)
    calls["agent"] = calls.agent.str.split(":").str[0]
    by = calls.groupby(["agent", "tier"]).agg(usd=("usd", "sum"), calls=("usd", "size"),
                                             tokens=("input_tokens", "sum"), p50_ms=("latency_ms", "median")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(by, x="agent", y="usd", color="tier", title="USD per agent"), width="stretch")
    stages = pd.DataFrame(lat["by_stage"].items(), columns=["stage", "ms"])
    b.plotly_chart(px.bar(stages, x="stage", y="ms", title="Critical-path latency by stage (ms)"),
                   width="stretch")
    st.dataframe(by, hide_index=True)
    st.subheader("Routing decisions")
    st.dataframe(routes, hide_index=True)
    st.subheader("OpenJev decisions")
    if not decisions.empty:
        st.dataframe(decisions.groupby(["task", "backend"]).agg(decisions=("task", "size"), items=("n_items", "sum"),
                                                                 mean_conf=("confidence", "mean"),
                                                                 escalated=("escalated", "sum")).reset_index(),
                     hide_index=True)

# ------------------------------------------------------------------ compare inputs
with tabs[6]:
    r = runs[~runs.variant.str.startswith("golden")].copy()
    order = {"small": 0, "medium": 1, "large": 2, "xlarge": 3}
    r = r[r["size"].isin(order)].assign(o=lambda x: x["size"].map(order)).sort_values("o")
    if not r.empty:
        sc = st.selectbox("Scenario", sorted(r.scenario.unique()))
        rr = r[r.scenario == sc]
        a, b = st.columns(2)
        a.plotly_chart(px.line(rr, x="size", y="total_usd", color="variant", markers=True,
                               title="Run cost (USD) by input size"), width="stretch")
        b.plotly_chart(px.line(rr, x="size", y="e2e_latency_ms", color="variant", markers=True,
                               title="End-to-end latency (ms) by input size"), width="stretch")
        st.dataframe(rr[["size", "variant", "n_candidates", "total_usd", "llm_calls", "input_tokens",
                         "e2e_latency_ms", "mean_eval", "hitl_pending"]], hide_index=True)

with tabs[7]:
    st.dataframe(calls, hide_index=True)