# Grit Steering Vector

An activation steering vector that adds **determination** to a language model:
it steers the residual stream so the model reframes failure as iteration and
continues trying rather than recommending that the user stop.

Trained on **Qwen3-0.6B** (`Qwen3ForCausalLM`) and applied at inference inside
the **vLLM** engine, on AMD Strix Halo (gfx1151).

## Method

1. **Train** a contrastive steering vector with PCA. Each pair uses the same
   challenge prompt in both halves; only the completion differs (determined vs
   defeatist). PCA on the pairwise-centered activations isolates the direction
   of maximum variance — the determination axis, independent of task content
   (`pca_pairwise`).
2. **Apply** the resulting vector inside vLLM via a worker-extension hook: add
   `coefficient * dir` to the residual stream at the chosen layer. Increasing
   the coefficient increases the effect.

## Dependencies

This project uses two libraries for vector training and inference-time
application:

- **activation-steering** — `SteeringVector.train`, the PCA-based contrastive
  vector extractor.
- **vLLM-Hook** — the `steer_hook_act` worker, applied via vLLM's
  `worker_extension_cls`.

## Why Qwen3-0.6B

Qwen3.5-0.8B is a hybrid Mamba/attention multimodal model
(`Qwen3_5ForConditionalGeneration`) that is not supported by current vLLM
builds ([#35391](https://github.com/vllm-project/vllm/issues/35391),
[#38041](https://github.com/vllm-project/vllm/issues/38041)). Qwen3-0.6B is a
vanilla dense transformer — fully supported by vLLM and a clean hook target
(its layers match the `model.layers.N` regex).

## Layout

```
.
├── data/grit_pairs.py          # contrastive dataset (determined vs defeatist pairs)
├── scripts/
│   ├── toolbox_setup.sh        # inside toolbox: install packages (--no-deps)
│   ├── train_grit_vector.py    # train .svec via pca_pairwise (transformers)
│   ├── make_hook_artifacts.py  # .svec -> .pt + vLLM-Hook config
│   └── test_grit.py            # apply vector at inference, grit ON vs OFF
└── README.md
```

## Pipeline

> **All steps run inside the `vllm` toolbox.** Installs use `--no-deps` so they
> never reinstall or downgrade the toolbox's torch / vLLM / ROCm stack, which is
> ABI-coupled and must not be touched.

```bash
toolbox enter vllm

# 1. install the steering stack WITHOUT touching the toolbox's core packages
bash grit/scripts/toolbox_setup.sh

# 2. download the model (skip if already cached in ~/.cache/huggingface)
huggingface-cli download Qwen/Qwen3-0.6B

# 3. train the grit vector (~1-2 min on gfx1151)
python grit/scripts/train_grit_vector.py

# 4. convert to vLLM-Hook format + write the model config
python grit/scripts/make_hook_artifacts.py

# 5. test — grit ON vs OFF on "should I give up?" prompts
python grit/scripts/test_grit.py --coefficient 4.0
```

## Contrastive dataset

`pca_pairwise` reads activations on the **completion suffix tokens**. Each pair
shares an identical challenge; only the completion differs, so the only signal
PCA can latch onto is the attitude, not the task. The first principal component
is the determination direction; the sign is set so determined activations
project positive. See [`data/grit_pairs.py`](data/grit_pairs.py).

## Tuning

`add_vector` adds `coefficient * dir` at the chosen layer. Sweep the
coefficient — too high degrades coherence; ~3-5 is a typical range for a 0.6B
model. The optimal layer is selected automatically by explained variance during
training (reported in `vectors/grit_manifest.json`).
