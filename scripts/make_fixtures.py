"""Record reference outputs from the Python `laya` SDK for the Rust parity tests.

Run once (needs `pip install laya` and the checkpoint):

    python scripts/make_fixtures.py [--model convaiinnovations/laya]

Writes tests/fixtures/cases.json (inputs, token ids, markers, and answers) and copies the
tokenizer files next to it so the sequence tests run without the full checkpoint.
"""
import argparse
import json
import os
import shutil

import laya
from laya.common import build_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")

CATEGORIES = {
    "Dining": "Restaurants, cafes, bars",
    "Groceries": "Supermarkets and food stores",
    "Gas": None,
    "Gas (2)": None,
    "Utilities › Electric": "Power company bills",
    "Café & Bakery": "Coffee shops, pâtisseries",
    "Credit Card Payment": "Payments to a card issuer",
    "Transfers": "Moves between own accounts",
    "other": None,
}

LONG_TEXT = " ".join(f"line {i} of a very long merchant memo with extra words" for i in range(80))

CASES = [
    {
        "name": "myphin_choice",
        "state": {"transactionTitle": "AMEX EPAYMENT ACH PMT", "direction": "money out"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which spending category does this bank transaction belong to? Pick other if none fits.",
                "criteria": CATEGORIES,
            }
        },
    },
    {
        "name": "text_state_with_mask_and_emoji",
        "state": "Paid at STARBUCKS #1234 [MASK] ☕ 😀 4.75",
        "questions": {
            "cat": {"type": "choice", "instructions": "Coffee or not?", "criteria": ["coffee", "not coffee"]}
        },
    },
    {
        "name": "many_options_and_long_state",
        "state": {"transactionTitle": LONG_TEXT, "direction": "money in"},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Pick one.",
                "criteria": {f"Option {i}": " ".join(["descriptive"] * 60) for i in range(15)},
            }
        },
    },
    {
        "name": "mixed_types",
        "state": {"subject": "Your invoice is overdue", "body": "Please pay within 3 days."},
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this message?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
            },
            "is_spam": {"type": "noul", "instructions": "Is this message spam?"},
            "is_billing": {
                "type": "noul",
                "instructions": "Is this about billing?",
                "criteria": {"false": "not about money", "true": "invoices or payments"},
            },
            "department": {
                "type": "choice",
                "instructions": {"question": "Which team?", "note": "résumé"},
                "criteria": {"billing": {"desc": "invoices", "n": 1}, "support": "help", "zero": 0},
            },
        },
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    args = ap.parse_args()

    agent = laya.load(args.model, device="cpu", subfolder=args.subfolder)
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    out = []
    for case in CASES:
        sequences = {}
        for qid, qdef in case["questions"].items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, case["state"], q, max_len, head_max_len)
            sequences[qid] = {"ids": ids, "markers": markers}
        answer = agent.system_one(case["state"], case["questions"])
        out.append({**case, "sequences": sequences, "answer": answer})
        print(case["name"], {k: (v.get("choice"), v.get("score"), v.get("noul"), v["confidence"]) for k, v in answer["answers"].items()})

    os.makedirs(os.path.join(FIXTURES, "tokenizer"), exist_ok=True)
    tok_dir = agent.tok.name_or_path
    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tok_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIXTURES, "tokenizer", name))
    with open(os.path.join(FIXTURES, "cases.json"), "w") as f:
        json.dump({"model": args.model, "max_len": max_len, "head_max_len": head_max_len, "cases": out}, f, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(FIXTURES, "cases.json"))


if __name__ == "__main__":
    main()
