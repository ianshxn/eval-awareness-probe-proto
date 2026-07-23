#!/usr/bin/env python
"""
Build a labelled eval-vs-deploy test set from the Needham et al. (2025)
Evaluation Awareness Benchmark ("Large Language Models Often Know When They Are
Being Evaluated", arXiv:2505.23836).

The dataset is GATED and ENCRYPTED -- this script does NOT download it. Obtain
dataset.json yourself first:

    # accept the terms at https://huggingface.co/datasets/jjpn2/eval_awareness
    huggingface-cli login
    git lfs install
    git clone https://huggingface.co/datasets/jjpn2/eval_awareness.git
    cd eval_awareness && ./scripts/decrypt.sh      # -> dataset.json

then:

    python scripts/build_needham_eval.py \
        --data /path/to/eval_awareness/dataset.json \
        --output datasets/needham_labelled.json

Do NOT commit the output to a public repo: the source terms forbid redistributing
GAIA content outside a gated/private location.

Ground-truth label: metadata.eval_category is true for evaluation (-> "testing")
and false for organic/real usage (-> "deployment").

Caveat -- the exact key that holds the transcript text is not documented in the
public README, so extract_text() below tries the common shapes and, if none
match, prints the record's keys so you can tell me the right field to finalize
the mapping. Agentic trajectories can be very long; on the 4k-context Phi-3 in
the colab notebook, prefer --scaffold chat and/or --max-chars to stay in window.
"""
import argparse
import json
import random


def eval_category(record):
    """True if this transcript is an evaluation (-> testing)."""
    meta = record.get("metadata", record)
    return bool(meta.get("eval_category"))


def scaffold_of(record):
    """'agentic' if metadata.scaffold is set, else 'chat'."""
    meta = record.get("metadata", record)
    return "chat" if meta.get("scaffold") in (None, "", "null") else "agentic"


def extract_text(record):
    """Flatten a Needham record to a single string. Provisional -- see module docstring."""
    # 1. A list of chat messages under a few likely keys.
    for key in ("transcript", "messages", "conversation", "prompt", "body", "text"):
        val = record.get(key)
        if isinstance(val, list):
            parts = []
            for m in val:
                if isinstance(m, dict):
                    role = m.get("role", m.get("from", ""))
                    content = m.get("content", m.get("value", ""))
                    parts.append(f"{role}: {content}".strip(": ").strip())
                else:
                    parts.append(str(m))
            return "\n\n".join(p for p in parts if p)
        if isinstance(val, str) and val.strip():
            return val
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build a labelled eval-vs-deploy test set from the Needham benchmark")
    parser.add_argument("--data", required=True,
                        help="Path to the decrypted dataset.json")
    parser.add_argument("--output", default="datasets/needham_labelled.json")
    parser.add_argument("--scaffold", choices=["all", "chat", "agentic"], default="all",
                        help="Keep only chat, only agentic, or all transcripts "
                             "(default all). Use 'chat' for short-context models.")
    parser.add_argument("--max-chars", type=int,
                        help="Truncate each transcript to this many characters "
                             "(keeps the tail). Helps fit short context windows.")
    parser.add_argument("--balance", action="store_true",
                        help="Downsample the larger class so AUROC is not skewed by base rate")
    parser.add_argument("--limit", type=int, help="Keep at most this many items (class-balanced)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        source = json.load(f)
    # dataset.json may be a list or a dict-of-records; normalise to a list.
    records = list(source.values()) if isinstance(source, dict) else source

    testing, deployment, skipped = [], [], 0
    for rec in records:
        if args.scaffold != "all" and scaffold_of(rec) != args.scaffold:
            continue
        text = extract_text(rec)
        if not text:
            if skipped == 0:
                print(f"Could not find a transcript field. Record keys were: "
                      f"{sorted(rec.keys())} -- tell me which holds the text.")
            skipped += 1
            continue
        if args.max_chars:
            text = text[-args.max_chars:]
        item = {"question": text, "type": "testing" if eval_category(rec) else "deployment"}
        (testing if item["type"] == "testing" else deployment).append(item)

    rng = random.Random(args.seed)
    if args.balance or args.limit:
        n = min(len(testing), len(deployment))
        rng.shuffle(testing)
        rng.shuffle(deployment)
        testing, deployment = testing[:n], deployment[:n]
    if args.limit:
        per = args.limit // 2
        testing, deployment = testing[:per], deployment[:per]

    labelled = testing + deployment
    rng.shuffle(labelled)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labelled, f, indent=2, ensure_ascii=True)

    print(f"Read {len(records)} records (scaffold filter: {args.scaffold})")
    if skipped:
        print(f"Skipped {skipped} records with no extractable transcript")
    print(f"Wrote {len(labelled)} labelled examples to {args.output} "
          f"({len(testing)} testing / {len(deployment)} deployment)")
    print("Reminder: do not commit this file to a public repo (GAIA redistribution terms).")


if __name__ == "__main__":
    main()
