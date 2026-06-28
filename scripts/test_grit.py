#!/usr/bin/env python3
"""Apply the grit steering vector at inference via vLLM-Hook.

Demonstrates the model WITH and WITHOUT the grit vector on prompts that
normally invite a "give up" answer. Designed to run inside the toolbox
where vLLM (ROCm) and the vLLM-Hook plugin are installed.

Usage:
    python scripts/test_grit.py [--coefficient N]

Sweep the coefficient to dial grit up (higher = more relentless). The
'add_vector' method adds  coefficient * dir  to the residual at the chosen
layer; 0 == no steering (baseline).
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
GRIT_ROOT = os.path.dirname(HERE)
STEER_ROOT = os.path.dirname(GRIT_ROOT)
HOOK_ROOT = os.path.join(STEER_ROOT, "vLLM-Hook")
sys.path.insert(0, GRIT_ROOT)
# vllm_hook_plugins is expected to be `pip install -e`'d; fall back to sibling.
try:
    import vllm_hook_plugins  # noqa: F401
except ImportError:
    sys.path.insert(0, HOOK_ROOT)

# vLLM-Hook spawns worker subprocesses; set spawn BEFORE importing vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM

MODEL_ID = "Qwen/Qwen3-0.6B"
CONFIG = os.path.join(HOOK_ROOT, "model_configs", "activation_steer", "Qwen3-0.6B.json")

# Prompts that tempt a model to give up / recommend quitting.
QUIT_PROMPTS = [
    "I've been debugging this for three hours and it still doesn't work. Should I just give up?",
    "I failed my driving test for the second time. Maybe I'm just not cut out for driving?",
    "My business lost money again this quarter. Is it time to close up shop?",
    "I can't seem to lose weight no matter what I try. Should I just accept I'll always be overweight?",
    "I've applied to fifty jobs and been rejected by all of them. Should I stop trying?",
    "This math proof is impossible. I've worked on it all week with no progress.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coefficient", type=float, default=4.0,
                    help="steering strength (higher = more grit). 0 = baseline.")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    with open(CONFIG) as f:
        cfg = json.load(f)
    default_steer = cfg["steering"]
    print(f"Layer: {default_steer['optimal_layer']}  "
          f"base coefficient (will override): {default_steer['coefficient']}")

    llm = HookLLM(
        model=MODEL_ID,
        worker_name="steer_hook_act",
        config_file=CONFIG,
        download_dir=os.path.expanduser("~/.cache"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    grit_on = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens,
        extra_args={"steer": {**default_steer, "method": "add_vector",
                              "coefficient": args.coefficient}},
    )
    grit_off = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print(f"\n{'='*70}\nGRIT STEERING TEST — coefficient={args.coefficient}\n{'='*70}")
    for prompt in QUIT_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        text = llm.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

        llm.llm_engine.reset_prefix_cache()
        off = llm.generate(text, grit_off, use_hook=False)[0].outputs[0].text

        llm.llm_engine.reset_prefix_cache()
        on = llm.generate(text, grit_on, use_hook=True)[0].outputs[0].text

        print(f"\n{'─'*70}\nPROMPT: {prompt}")
        print(f"\n[GRIT OFF — baseline]:\n{off.strip()}")
        print(f"\n[GRIT ON  — coef {args.coefficient}]:\n{on.strip()}")


if __name__ == "__main__":
    main()
