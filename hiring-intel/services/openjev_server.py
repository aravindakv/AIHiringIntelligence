"""OpenJev as a small HTTP service, so the app containers stay light (no torch) and the model loads once.

    uvicorn services.openjev_server:app --port 8200
    POST /entail  {"premise": "...", "hypotheses": ["...", "..."]}  -> {"entailment": [0.91, 0.04]}
    POST /decide  {"premise": "...", "question": "...", "options": ["yes", "no"]} -> Decision
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_intel.openjev import CFG, LocalOpenJev, OpenJevDecider  # noqa: E402

app = FastAPI(title="OpenJev decision service")
_model = LocalOpenJev(os.getenv("OPENJEV_REPO", CFG["repo"]), os.getenv("OPENJEV_SUBFOLDER", CFG["subfolder"]))
_decider = OpenJevDecider(_model)


class EntailReq(BaseModel):
    premise: str
    hypotheses: list[str]


class DecideReq(BaseModel):
    premise: str
    question: str
    options: list[str]
    task: str = "api"


@app.get("/health")
def health():
    return {"ok": True, "device": _model.device, "subfolder": os.getenv("OPENJEV_SUBFOLDER", CFG["subfolder"])}


@app.post("/entail")
def entail(r: EntailReq):
    return {"entailment": _model.entail(r.premise, r.hypotheses)}


@app.post("/decide")
def decide(r: DecideReq):
    return _decider.choose(r.task, r.premise, r.question, r.options).model_dump()