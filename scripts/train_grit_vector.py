#!/usr/bin/env python3
"""Train the grit/perseverance steering vector for Qwen3-0.6B.

Pipeline:
  1. Load Qwen3-0.6B in bf16 on the GPU (ROCm/HIP via PyTorch).
  2. Build a SteeringDataset from data/grit_pairs.py — identical challenge
     in both halves, contrasting gritty/quitter completions as suffixes.
  3. Train per-layer directions via pca_pairwise (recommended default).
  4. Save .svec, then rank layers by explained variance and print the top
     candidates so we can choose the best layer for the vLLM-Hook config.

Usage (run INSIDE the toolbox, where torch+ROCm and the model live):
    python scripts/train_grit_vector.py

Outputs:
    grit/vectors/grit_qwen3-0.6b.svec
    grit/figures/grit_variance_by_layer.png
"""
import os
import sys
import json

# Make the grit package + the cloned activation-steering importable.
HERE = os.path.dirname(os.path.abspath(__file__))
GRIT_ROOT = os.path.dirname(HERE)            # .../grit
sys.path.insert(0, GRIT_ROOT)                # for `from data.grit_pairs import ...`
# activation_steering is expected to be `pip install -e`'d (preferred), but
# fall back to a sibling checkout for dev convenience.
try:
    import activation_steering  # noqa: F401
except ImportError:
    STEER_ROOT = os.path.dirname(GRIT_ROOT)
    sys.path.insert(0, os.path.join(STEER_ROOT, "activation-steering"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steering import SteeringDataset, SteeringVector
from data.grit_pairs import get_examples

MODEL_ID = "Qwen/Qwen3-0.6B"
# Cache location the toolbox/HF share with the host. We already downloaded
# the snapshot to the hub dir, so this resolves locally with no network.
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
OUT_DIR = os.path.join(GRIT_ROOT, "vectors")
FIG_DIR = os.path.join(GRIT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("GRIT VECTOR TRAINING — Qwen3-0.6B")
    print("=" * 60)

    # --- 1. Load model + tokenizer ---
    print(f"\n[1/4] Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 is the natural dtype for this model and fast on gfx1151.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"      device={next(model.parameters()).device} "
          f"dtype={next(model.parameters()).dtype} "
          f"layers={model.config.num_hidden_layers} "
          f"hidden={model.config.hidden_size}")

    # --- 2. Build dataset ---
    print("\n[2/4] Building grit contrastive dataset ...")
    examples, suffixes = get_examples()
    print(f"      {len(examples)} challenge prompts x {len(suffixes)} suffix pairs")
    dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=examples,
        suffixes=suffixes,
        use_chat_template=True,
    )

    # --- 3. Train vector (pca_pairwise over all layers) ---
    print("\n[3/4] Training steering vector (pca_pairwise) ...")
    sv = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=dataset,
        method="pca_pairwise",
        accumulate_last_x_tokens="suffix-only",  # read activations on the completion tokens
        save_analysis=False,
        output_dir=FIG_DIR,
    )

    # --- 4. Save + rank layers ---
    svec_path = os.path.join(OUT_DIR, "grit_qwen3-0.6b.svec")
    sv.save(svec_path)
    print(f"\n[4/4] Saved steering vector -> {svec_path}")

    print("\n" + "=" * 60)
    print("EXPLAINED VARIANCE BY LAYER (higher = grit/quit is cleaner here)")
    print("=" * 60)
    ranked = sorted(sv.explained_variances.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'layer':>6}  {'explained_var':>14}")
    for layer, var in ranked:
        marker = "  <-- BEST" if layer == ranked[0][0] else ""
        print(f"{layer:>6}  {var:>14.4f}{marker}")

    # Persist a small manifest the converter + config writer can read.
    manifest = {
        "model_id": MODEL_ID,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "method": "pca_pairwise",
        "svec_path": svec_path,
        "best_layer": int(ranked[0][0]),
        "best_layer_variance": float(ranked[0][1]),
        "top5_layers": [{"layer": int(l), "variance": float(v)} for l, v in ranked[:5]],
        "explained_variances": {int(k): float(v) for k, v in sv.explained_variances.items()},
    }
    manifest_path = os.path.join(OUT_DIR, "grit_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {manifest_path}")
    print(f"Recommended layer for vLLM-Hook config: {ranked[0][0]} "
          f"(var {ranked[0][1]:.4f})")


if __name__ == "__main__":
    main()
