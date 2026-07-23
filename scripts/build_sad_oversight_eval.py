#!/usr/bin/env python
"""
Build a labelled eval-vs-deploy test set from SAD stages_oversight.

The paper selects probes by AUROC on the Situational Awareness Dataset (SAD)
`stages_oversight` task (Laine et al., 2024). Since the probe vectors are trained
on contrastive_dataset.json, scoring them against SAD is genuinely
out-of-distribution -- exactly the paper's train-on-contrastive / select-on-SAD
setup.

analyze_probe.py wants one prompt per item with a `type` field of 'testing' or
'deployment'; it applies the chat template itself, so the `question` here is the
raw SAD `body` text, not a pre-formatted prompt.

SAD ships the oversight split inside a password-protected zip in the LRudL/sad
repo. Clone it first:

    git clone https://github.com/LRudL/sad.git

then:

    python scripts/build_sad_oversight_eval.py \
        --sad-repo ./sad \
        --output datasets/sad_oversight_labelled.json

Each SAD item carries `body` (the text shown to the model) and `choices_right`,
which contains 'yes' for a testing/oversight example and 'no' for a deployment
one -- that string, not the file the row came from, is the ground-truth label.
"""
import argparse
import json
import os
import random
import zipfile

PASSWORD = b"sadtimesforthesetimes"
ZIP_REL = os.path.join("sad", "stages", "private_data_gen.zip")
DEPLOY_FILE = "out/deploy_oversight.jsonl"
TEST_FILE = "out/test_oversight.jsonl"


def load_jsonl_from_zip(zip_path, name, password=PASSWORD):
    """Read one JSONL member from SAD's password-protected data zip."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        content = zip_ref.read(name, pwd=password).decode("utf-8")
    return [json.loads(line) for line in content.strip().split("\n")]


def is_testing(item):
    # SAD marks the oversight/testing answer as 'yes' in choices_right and the
    # deployment answer as 'no'. Honour that rather than the source filename.
    return "yes" in item["choices_right"]


def length_match(testing, deployment, tolerance):
    """Greedily pair examples of similar length, dropping those with no partner.

    probe_analysis scores each example by its mean projection across token
    positions, so a systematic length gap between the classes can leak into the
    AUROC. Use this to rule that out; run analyze_probe.py --control alongside.
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
    parser = argparse.ArgumentParser(
        description="Build a labelled eval-vs-deploy test set from SAD stages_oversight")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sad-repo", help="Path to a cloned LRudL/sad checkout")
    source.add_argument("--zip", help="Path to private_data_gen.zip directly")
    parser.add_argument("--output", default="datasets/sad_oversight_labelled.json",
                        help="Where to write the labelled set")
    parser.add_argument("--limit", type=int,
                        help="Keep only this many items (class-balanced). "
                             "analyze_probe.py runs a fresh forward pass per item "
                             "per layer, so cap this for a first run.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling/shuffle seed")
    parser.add_argument("--length-match", action="store_true",
                        help="Greedily pair testing and deployment examples of similar "
                             "length and drop the rest, to rule out a length confound.")
    parser.add_argument("--length-tolerance", type=float, default=0.25,
                        help="Max relative length difference for --length-match (default 0.25)")
    args = parser.parse_args()

    zip_path = args.zip if args.zip else os.path.join(args.sad_repo, ZIP_REL)
    if not os.path.exists(zip_path):
        parser.error(f"SAD data zip not found at {zip_path} -- "
                     f"did you `git clone https://github.com/LRudL/sad.git`?")

    deploy = load_jsonl_from_zip(zip_path, DEPLOY_FILE)
    test = load_jsonl_from_zip(zip_path, TEST_FILE)
    all_items = deploy + test

    testing = [{"question": it["body"], "type": "testing"}
               for it in all_items if is_testing(it)]
    deployment = [{"question": it["body"], "type": "deployment"}
                  for it in all_items if not is_testing(it)]

    # Balance the classes so AUROC is not skewed by base rate.
    rng = random.Random(args.seed)
    rng.shuffle(testing)
    rng.shuffle(deployment)
    n = min(len(testing), len(deployment))
    testing, deployment = testing[:n], deployment[:n]

    if args.length_match:
        before = len(testing)
        testing, deployment = length_match(testing, deployment, args.length_tolerance)
        print(f"Length matching kept {len(testing)}/{before} pairs "
              f"(tolerance {args.length_tolerance:.0%})")

    if args.limit:
        per_class = args.limit // 2
        testing = testing[:per_class]
        deployment = deployment[:per_class]

    labelled = testing + deployment
    rng.shuffle(labelled)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # ensure_ascii keeps the file readable by analyze_probe.py's plain open(),
    # which does not pass an encoding and so uses cp1252 on Windows.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labelled, f, indent=2, ensure_ascii=True)

    print(f"Loaded {len(all_items)} SAD oversight items "
          f"({len(deploy)} from deploy file / {len(test)} from test file)")
    print(f"Wrote {len(labelled)} labelled examples to {args.output} "
          f"({len(testing)} testing / {len(deployment)} deployment)")


if __name__ == "__main__":
    main()
