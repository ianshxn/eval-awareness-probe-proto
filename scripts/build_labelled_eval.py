#!/usr/bin/env python
"""
Build a labelled eval-vs-deploy test set for analyze_probe.py.

The shipped datasets store examples in contrastive form: one `question` holding
both texts as "(A) ... \n(B) ...", plus `positive`/`negative` marking which side
is the testing one. analyze_probe.py instead wants one prompt per item with a
`type` field of 'testing' or 'deployment' -- without it every example falls
through to 'deployment', the ROC gets a single class and the AUC is undefined.

tom_vs_sharegpt.json is the right source: the (A) side is a theory-of-mind
benchmark item (testing), the (B) side is a real ShareGPT user turn (deployment).
Splitting each item in two gives a labelled set with the A/B position confound
removed, since the label follows the content rather than the slot.

Usage:
    python scripts/build_labelled_eval.py \
        --data datasets/tom_vs_sharegpt.json \
        --output datasets/eval_vs_deploy_labelled.json
"""
import argparse
import json
import random

PREFIX = "Choices:\n\n(A) "
SPLIT = "\n(B) "


def split_item(entry):
    """Split one contrastive entry into (testing_text, deployment_text)."""
    question = entry["question"]
    if SPLIT not in question:
        return None

    a_side, b_side = question.split(SPLIT, 1)
    if a_side.startswith(PREFIX):
        a_side = a_side[len(PREFIX):]

    # `positive` names the side that is the testing example; it is "(A)" for
    # every item in tom_vs_sharegpt, but honour it rather than assume it.
    if entry.get("positive", "(A)").strip() == "(A)":
        return a_side.strip(), b_side.strip()
    return b_side.strip(), a_side.strip()


def length_match(testing, deployment, tolerance):
    """Greedily pair examples of similar length, dropping those with no partner.

    Both lists are walked in length order; a testing example is kept only if the
    next unused deployment example is within `tolerance` relative length of it.
    """
    testing = sorted(testing, key=lambda x: len(x["question"]))
    deployment = sorted(deployment, key=lambda x: len(x["question"]))

    kept_test, kept_deploy = [], []
    i = j = 0
    while i < len(testing) and j < len(deployment):
        len_t, len_d = len(testing[i]["question"]), len(deployment[j]["question"])
        if abs(len_t - len_d) / max(len_t, len_d) <= tolerance:
            kept_test.append(testing[i])
            kept_deploy.append(deployment[j])
            i += 1
            j += 1
        elif len_d < len_t:
            j += 1
        else:
            i += 1

    return kept_test, kept_deploy


def main():
    parser = argparse.ArgumentParser(description="Build a labelled eval-vs-deploy test set")
    parser.add_argument("--data", default="datasets/tom_vs_sharegpt.json",
                        help="Contrastive source dataset")
    parser.add_argument("--output", default="datasets/eval_vs_deploy_labelled.json",
                        help="Where to write the labelled set")
    parser.add_argument("--limit", type=int,
                        help="Keep only this many items (class-balanced, sampled "
                             "with --seed). analyze_probe.py runs a fresh forward "
                             "pass per item per layer, so cap this for a first run.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--min-chars", type=int, default=1,
                        help="Drop a pair if either side is shorter than this")
    parser.add_argument("--length-match", action="store_true",
                        help="Greedily pair testing and deployment examples of similar "
                             "length and drop the rest. The raw split has a severe length "
                             "confound (ToM items run ~5x longer than ShareGPT turns), and "
                             "probe_analysis scores each example by its mean projection "
                             "across token positions, so length alone separates the classes "
                             "at AUROC ~0.82. Use this to check the probe beats that.")
    parser.add_argument("--length-tolerance", type=float, default=0.25,
                        help="Max relative length difference for --length-match (default 0.25)")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        source = json.load(f)

    testing, deployment = [], []
    skipped = 0
    for entry in source:
        pair = split_item(entry)
        if pair is None:
            skipped += 1
            continue
        test_text, deploy_text = pair
        # Drop the pair if either side is blank, so the two classes stay balanced.
        if len(test_text) < args.min_chars or len(deploy_text) < args.min_chars:
            skipped += 1
            continue
        testing.append({"question": test_text, "type": "testing"})
        deployment.append({"question": deploy_text, "type": "deployment"})

    if args.length_match:
        before = len(testing)
        testing, deployment = length_match(testing, deployment, args.length_tolerance)
        print(f"Length matching kept {len(testing)}/{before} pairs "
              f"(tolerance {args.length_tolerance:.0%})")

    if args.limit:
        per_class = args.limit // 2
        rng = random.Random(args.seed)
        testing = rng.sample(testing, min(per_class, len(testing)))
        deployment = rng.sample(deployment, min(per_class, len(deployment)))

    labelled = testing + deployment
    random.Random(args.seed).shuffle(labelled)

    # ensure_ascii keeps the file readable by analyze_probe.py's plain open(),
    # which does not pass an encoding and so uses cp1252 on Windows.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labelled, f, indent=2, ensure_ascii=True)

    print(f"Read {len(source)} contrastive items from {args.data}")
    if skipped:
        print(f"Skipped {skipped} items with no '{SPLIT.strip()}' delimiter")
    print(f"Wrote {len(labelled)} labelled examples to {args.output} "
          f"({len(testing)} testing / {len(deployment)} deployment)")


if __name__ == "__main__":
    main()
