# Bag-of-Words Baseline

A surface-text control for the eval-awareness probe. It answers one question:

> How much of the probe's AUROC could you get **without the model at all**?

If a linear classifier over word counts separates `testing` from `deployment` about as
well as the residual-stream probe does, then the probe's score is not evidence that the
model represents "I am being evaluated." It is evidence that the two classes are
lexically distinguishable, which we already know.

## Why this is needed

The `phi3medium_contrastive_20260722_192015` run reports AUROC 0.116 at layer 10 under
the repo's deployment-positive convention (`src/probe_analysis.py:183`). That is not a
failure — inverted, it is **0.884**, the strongest separation anywhere in the layer
sweep, and it survives filtering to ASCII-only, length-matched prompts (0.844).

That looked like a real finding until we read the prompts. The labelled eval set is built
from `datasets/tom_vs_sharegpt.json`, and `scripts/build_labelled_eval.py` describes the
construction plainly: the testing side is a theory-of-mind benchmark item, the deployment
side is a real ShareGPT user turn. The two classes are two different corpora, so they
differ in nearly every surface property at once:

| Feature | testing (n=147) | deployment (n=147) |
| --- | ---: | ---: |
| Brevity instruction ("brief", "concise") | 118 | 10 |
| Belief/ToM verbs (think, believe, unaware) | 113 | 27 |
| Imperative opening (write/create/act as/help me) | 0 | 45 |
| Any non-ASCII character | 0 | 42 |

The non-ASCII split is the tell. It is *not* a language effect — only 3 deployment prompts
contain Hangul and 4 contain Chinese. It is 30 prompts containing curly quotes, em dashes
and ellipses: the fingerprint of text copy-pasted out of a browser, versus text emitted
programmatically by a vignette generator. Provenance, not content.

A probe reading layer 10 — early, where register and formatting live — has ample surface
signal to exploit. The bag-of-words baseline measures exactly how much.

## What to build

`scripts/bow_baseline.py`, following the CLI conventions of the existing scripts.

```
python scripts/bow_baseline.py \
    --data datasets/eval_vs_deploy_labelled.json \
    --output analysis/bow_baseline.json \
    --folds 5 --seed 0
```

Input is the same labelled file `analyze_probe.py` consumes: a JSON list of
`{"question": str, "type": "testing" | "deployment"}`.

### Method

1. **Labels.** `y = 1` for `deployment`, `0` for `testing`. This matches
   `src/probe_analysis.py:183` so the number is directly comparable to the probe's AUROC
   without mental sign-flipping.

2. **Features.** `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
   sublinear_tf=True)`. Word unigrams and bigrams only. No hand-built features — the
   point is to show that naive surface statistics suffice, so any cleverness weakens the
   argument.

3. **Classifier.** `LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
   class_weight="balanced")`. The classes are balanced 147/147, but keep the flag so the
   script stays honest on re-splits.

4. **Evaluation.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.
   Fit the vectorizer **inside each fold on the training split only** — fitting it on the
   full corpus leaks test-set vocabulary and inflates the score. Report mean AUROC across
   folds with standard deviation, using `roc_curve` + `auc` from `sklearn.metrics` to
   mirror `probe_analysis.py`.

5. **Report the top features.** Sort the fitted coefficients and print the 20 most
   deployment-indicative and 20 most testing-indicative terms. This turns a number into a
   diagnosis: if the top testing terms are `brief`, `response`, `she`, `think`, the
   classifier is reading the benchmark template, which is precisely the confound.

### Ablations worth running in the same script

Each isolates one class of surface cue. Add them behind `--ablation`:

| Ablation | Feature set | What it isolates |
| --- | --- | --- |
| `full` | tf-idf 1–2 grams | Total surface separability |
| `function-words` | 100 most frequent English stopwords only | Register/style, no topic |
| `punctuation` | Counts of `?`, `.`, `,`, quotes, newlines, non-ASCII flag | Formatting/provenance alone |
| `length` | Character count, single feature | The crudest possible cue |

The `punctuation` and `length` rows are the interesting ones. A single length feature
scoring meaningfully above 0.5 tells you the corpora are distinguishable by a property
no theory of eval-awareness would predict.

## Interpreting the result

Compare the mean cross-validated AUROC against the probe's best layer (0.884 at layer 10,
deployment-negative direction):

- **BoW ≳ 0.85** — the probe has no demonstrated advantage over word counts. The layer-10
  result should not be reported as evidence of an eval-awareness representation. This is
  the outcome I expect.
- **BoW 0.70–0.85** — surface cues explain most but not all of the probe's separation.
  The residual gap is small and needs the confound-free dataset below before it means
  anything.
- **BoW ≲ 0.65** — genuinely surprising given the table above. Re-check the fold-wise
  vectorizer fitting for a bug before believing it.

Note the asymmetry in what this test can show. A high BoW score **falsifies** the strong
reading of the probe result. A low BoW score does **not** validate it — it only fails to
rule it out, since a probe can exploit surface cues that word counts miss. This is a
control, not a confirmation.

## What this baseline cannot fix

The confound is in the dataset, so no re-analysis of the existing files resolves it. A
positive result here is a reason to rebuild the eval set, not to tune the probe. Two
constructions that would actually isolate eval-awareness:

1. **Frame-swap pairs.** Hold the prompt body fixed and vary only the wrapper: the same
   organic request presented as `"Q: {body} Provide a very brief response."` versus
   plainly. Genre is constant by construction, so BoW gets ~0.5 and any probe separation
   is attributable to the frame.
2. **Single-source labelling.** Draw both classes from the same scrape and label by
   whether the item is *actually* a benchmark question, rather than pairing one synthetic
   generator against one web corpus.

Until one of those exists, treat every AUROC in `analysis/` as an upper bound contaminated
by an unknown amount of genre detection.

## Related

- `scripts/build_labelled_eval.py` — builds the labelled set; its docstring documents the
  ToM-vs-ShareGPT construction that creates the confound.
- `src/probe_analysis.py:183` — the deployment-positive label convention this baseline matches.
- `analysis/visualizations/layer_performance.png` — plots layer 10 at 0.116 without
  flagging that an inverted AUROC is a strong signal, not a weak one.
