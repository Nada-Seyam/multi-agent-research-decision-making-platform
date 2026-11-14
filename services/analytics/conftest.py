import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _DeterministicNLI:
    """Small semantic test double; production always uses the real model."""

    def entailment_probabilities(self, pairs):
        probabilities = []
        for premise, hypothesis in pairs:
            premise = premise.lower()
            supports = "supports choosing" in hypothesis.lower()
            if "robust but costly" in premise:
                probability = 0.8
            elif "does not create a reliability risk" in premise:
                probability = 0.9 if supports else 0.05
            elif any(
                phrase in premise
                for phrase in (
                    "recommend",
                    "superior",
                    "meets the constraint",
                    "meets the latency constraint",
                )
            ):
                probability = 0.9 if supports else 0.05
            elif any(
                phrase in premise
                for phrase in (
                    "fails the reliability",
                    "critical weakness",
                    "totally infeasible",
                    "should not be chosen",
                )
            ):
                probability = 0.05 if supports else 0.9
            else:
                probability = 0.05
            probabilities.append(probability)
        return probabilities


@pytest.fixture(autouse=True)
def _mock_nli_model(monkeypatch):
    """Unit tests must not download or execute a 200M-parameter model."""
    from analytics import sentiment

    classifier = _DeterministicNLI()
    monkeypatch.setattr(sentiment, "_get_nli_classifier", lambda: classifier)
