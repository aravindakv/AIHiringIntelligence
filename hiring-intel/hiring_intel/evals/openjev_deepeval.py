"""OpenJev as a DeepEval "System One" model.

DeepEval 4.x metrics such as Faithfulness and Hallucination accept `eval_mode="system_one"` (or
"hybrid") plus a `system_one_model`: a model that answers typed questions (yes/no "noul", choice,
score) with probabilities instead of generating text. That is exactly what OpenJev does, so this
adapter lets DeepEval judge with OpenJev locally at $0 API cost.

    from deepeval.metrics import FaithfulnessMetric
    m = FaithfulnessMetric(eval_mode="system_one", system_one_model=OpenJevSystemOne(), include_reason=False)
"""
from __future__ import annotations

import json
from typing import Any

from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion,
                                               ScoreAnswer, ScoreQuestion, SystemOneAnswers)

from hiring_intel.openjev import OpenJevDecider


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)


class OpenJevSystemOne(DeepEvalBaseSystemOneModel):
    def __init__(self, decider: OpenJevDecider | None = None):
        self._decider = decider
        super().__init__(model="openjev")

    def load_model(self, *args, **kwargs):
        return self._decider or OpenJevDecider()

    def get_model_name(self, *args, **kwargs) -> str:
        return f"openjev ({self.model.backend.name})"

    def decide(self, state: Any, questions: dict) -> tuple[SystemOneAnswers, float | None]:
        premise = _text(state)
        ans = SystemOneAnswers()
        for key, q in questions.items():
            instr = _text(q.instructions)
            if isinstance(q, ChoiceQuestion):
                opts = list(q.options)
                d = self.model.choose(f"deepeval:{key}", premise, instr, opts)
                ans.choices[key] = ChoiceAnswer(choice=d.choice, probabilities=d.probs, confidence=d.confidence)
            elif isinstance(q, ScoreQuestion):
                levels = [str(l) for l in q.levels]
                d = self.model.choose(f"deepeval:{key}", premise, instr, levels)
                probs = {i + 1: d.probs[l] for i, l in enumerate(levels)}
                ans.scores[key] = ScoreAnswer(score=levels.index(d.choice) + 1, probabilities=probs,
                                              confidence=d.confidence)
            else:                                    # NoulQuestion: probability of "yes"
                d = self.model.yes_no(f"deepeval:{key}", premise, instr)
                ans.nouls[key] = NoulAnswer(probability=d.probs["yes"])
        return ans, 0.0                              # $0 API cost

    async def a_decide(self, state: Any, questions: dict):
        return self.decide(state, questions)