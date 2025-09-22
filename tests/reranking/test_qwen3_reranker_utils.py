import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen3_reranker_utils import extract_yes_probability


def test_extract_probability_from_logprobs():
    response = {
        "logprobs": [
            {
                "token": " yes",
                "logprob": math.log(0.8),
                "top_logprobs": [
                    {"token": " yes", "logprob": math.log(0.8)},
                    {"token": " no", "logprob": math.log(0.2)},
                ],
            }
        ]
    }

    probability = extract_yes_probability(response)
    assert math.isclose(probability, 0.8, rel_tol=1e-5)


def test_extract_probability_from_text_yes():
    response = {
        "response": "Yes, this matches",
    }

    probability = extract_yes_probability(response)
    assert probability > 0.9


def test_extract_probability_from_text_no():
    response = {
        "response": "No, it does not",
    }

    probability = extract_yes_probability(response)
    assert probability < 0.1
