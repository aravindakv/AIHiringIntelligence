from hiring_intel.graph import resume, run


def test_run_survives_crash_and_retries():
    s = run("data/raw/baseline_small", variant="t", crash_agent="offers", inject_fault="sourcing")
    assert "offers" in s["retried"] and "sourcing" in s["retried"]
    assert all(o["status"] != "failed" for o in s["outputs"].values())   # recovered on retry


def test_hitl_interrupt_and_resume():
    s = run("data/raw/baseline_small", variant="t", hitl_mode="interrupt")
    pending = s["__interrupt__"][0].value["pending"]
    assert pending
    s2 = resume(s["run_id"], {p["item_id"]: {"action": "rejected", "reviewer": "test"} for p in pending})
    assert {r["status"] for r in s2["review"]} <= {"rejected", "auto_published"}


def test_healthy_input_raises_nothing_high_at_scale():
    s = run("data/raw/healthy_small", variant="t")
    assert s["evals"] and all(e["overall"] > 0 for e in s["evals"].values())