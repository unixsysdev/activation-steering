#!/usr/bin/env python3
"""Convert the trained .svec grit vector into the artifacts vLLM-Hook needs.

vLLM-Hook's SteerHookActWorker loads a .pt dict with two keys:
    {"dir": Tensor[hidden_size], "avg_proj": Tensor scalar}

  - dir       : the steering direction at the chosen layer (grit positive)
  - avg_proj  : mean projection of the *un-steered* activations onto dir,
                used by the "adjust_rs" method (residual subtraction) to
                compute how much to add to reach the gritty baseline.

This script:
  1. Loads grit_manifest.json (best layer + svec path from training).
  2. Recomputes avg_proj by re-reading activations on the grit suffixes
     at the chosen layer, projecting onto dir. (We need the raw model
     activations, which the .svec alone doesn't store.)
  3. Writes grit_qwen3-0.6b.pt into vLLM-Hook/steering_vectors/.
  4. Writes the vLLM-Hook model config into vLLM-Hook/model_configs/.

Run inside the toolbox:
    python scripts/make_hook_artifacts.py [--layer N]
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
GRIT_ROOT = os.path.dirname(HERE)
STEER_ROOT = os.path.dirname(GRIT_ROOT)
sys.path.insert(0, GRIT_ROOT)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.grit_pairs import get_examples

MANIFEST = os.path.join(GRIT_ROOT, "vectors", "grit_manifest.json")
HOOK_ROOT = os.path.join(STEER_ROOT, "vLLM-Hook")
VEC_OUT = os.path.join(HOOK_ROOT, "steering_vectors", "grit_qwen3-0.6b.pt")
CFG_DIR = os.path.join(HOOK_ROOT, "model_configs", "activation_steer")
CFG_OUT = os.path.join(CFG_DIR, "Qwen3-0.6B.json")


def layer_hidden_state(model, tokenizer, text, layer):
    """Return the mean hidden state over the completion suffix tokens for one
    formatted (chat-templated, suffix-appended) string. Matches the
    accumulate_last_x_tokens='suffix-only' convention used in training."""
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # layer index in hidden_states tuple is layer+1 (index 0 is embeddings)
    hs = out.hidden_states[layer + 1]  # (1, seq, hidden)
    return hs[0].float().cpu()  # (seq, hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="override best layer from manifest")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    svec_path = manifest["svec_path"]
    layer = args.layer if args.layer is not None else manifest["best_layer"]
    model_id = manifest["model_id"]

    # Load svec
    with open(svec_path) as f:
        svec = json.load(f)
    dir_vec = np.array(svec["directions"][str(layer)], dtype=np.float32)
    print(f"Using layer {layer}  (var {svec['explained_variances'][str(layer)]:.4f})")
    print(f"dir shape: {dir_vec.shape}")

    # --- Recompute avg_proj from un-steered gritty activations ---
    print(f"\nLoading {model_id} to compute avg_proj ...")
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    examples, suffixes = get_examples()
    projections = []
    for (challenge, _), (gritty, _quitter) in zip(examples, suffixes):
        # Reproduce the same formatted string training used: chat template
        # with the user challenge, NO generation prompt, then the suffix.
        msg = [{"role": "user", "content": challenge}]
        base = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        text = base + gritty
        hs = layer_hidden_state(model, tokenizer, text, layer)
        # suffix-only: project the mean over the last len(suffix_tokens) tokens
        suf_tokens = tokenizer.encode(gritty, add_special_tokens=False)
        suf_hs = hs[-len(suf_tokens):, :]
        mean_hs = suf_hs.mean(dim=0).numpy()
        proj = float(mean_hs @ dir_vec / np.linalg.norm(dir_vec))
        projections.append(proj)

    avg_proj = float(np.mean(projections))
    print(f"avg_proj (gritty baseline) = {avg_proj:.4f}  over {len(projections)} pairs")

    # --- Write .pt ---
    os.makedirs(os.path.dirname(VEC_OUT), exist_ok=True)
    torch.save(
        {"dir": torch.tensor(dir_vec), "avg_proj": torch.tensor(avg_proj)},
        VEC_OUT,
    )
    print(f"\nWrote vector -> {VEC_OUT}")

    # --- Write vLLM-Hook config ---
    os.makedirs(CFG_DIR, exist_ok=True)
    rel_vec = os.path.relpath(VEC_OUT, HOOK_ROOT)
    cfg = {
        "model_info": {
            "provider": "attn-hf",
            "name": "qwen3-0.6b-grit",
            "model_id": model_id,
            "note": "Grit/perseverance steering vector for Qwen3-0.6B. "
                    "Trained via pca_pairwise on contrastive gritty-vs-quitter "
                    "completions. Applying this vector adds determination."
        },
        "steering": {
            "method": "add_vector",      # simplest robust application: add coef*dir
            "coefficient": 1,            # tunable at request time via extra_args
            "optimal_layer": int(layer),
            "vector_path": rel_vec,
            "apply_at_all_positions": True
        }
    }
    with open(CFG_OUT, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Wrote config  -> {CFG_OUT}")
    print("\nDone. Use the config with vLLM-Hook's steer_hook_act worker.")


if __name__ == "__main__":
    main()
