Project: Probing and steering evaluation awareness. Training probes on llama 3.3 70b instruct to see whether there's internal distinction between test vs deploy prompts, and steering with SAE features to detect sandbagging. This repo is a minimal implementation of probe training and evaluation.

Paper: https://arxiv.org/abs/2507.01786

Results: Linear probes trained with simple contrastive data generalises to distinguishing real eval and deploy prompts. Steering with this direction has no causal effects, but with SAE features has some effects on showing sandbagging.




**Usage**

1. Clone the repo
2. Get packages ```pip install -r requirements.txt```
3. Generate probes ```python scripts/generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR```
4. Build a labelled test set ```python scripts/build_labelled_eval.py --length-match```
5. Visualise ```python scripts/analyze_probe.py --model MODEL_PATH --vectors VECTORS_DIR --data datasets/eval_vs_deploy_labelled.json --output RESULTS_DIR --visualize```

Step 4 is not optional. `analyze_probe.py` labels examples by a `type` field that none of
the shipped datasets carry, so pointing it at one of them puts every example in a single
class and leaves the AUC undefined — the plots still render. `build_labelled_eval.py`
derives a labelled set from `tom_vs_sharegpt.json`; `--length-match` controls a length
confound that otherwise separates the classes at AUROC 0.82 on its own.

**Running on one GPU.** The 70B run above needs ~140GB in bf16 and TransformerLens has no
quantized load path, so it wants ~4x A100-80GB with `n_devices=4`. For a single 40GB A100,
`notebooks/colab_a100_run.ipynb` runs the same pipeline on `microsoft/Phi-3-medium-4k-instruct`
and documents the setup. Use the **4k** variant, not 128k — see that notebook for why.
