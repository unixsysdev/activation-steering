# Patches — dependency fixes for vLLM v0.22

This project depends on two upstream libraries that needed patching to run on
the toolbox's vLLM build (v0.22.1rc1). The patches below are against the
current upstream `main` of each repo and **apply cleanly** (`git apply --check`
verified). They are kept here as diffs rather than forks so the upstream stays
trackable.

All patches are local compatibility fixes — none change the algorithms.

---

## Upstream repos

| Patch | Applies to | Repo |
|---|---|---|
| `activation-steering_steering_vector.py.diff` | `activation_steering/steering_vector.py` | https://github.com/IBM/activation-steering |
| `vllm-hook_hook_llm.py.diff` | `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py` | https://github.com/IBM/vLLM-Hook |
| `vllm-hook_steer_activation_worker.py.diff` | `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py` | https://github.com/IBM/vLLM-Hook |

## Applying

From the **root of each cloned upstream repo**, run the relevant `git apply`:

```bash
# activation-steering (cloned at $STEER_ROOT/activation-steering)
cd activation-steering
git apply grit/patches/activation-steering_steering_vector.py.diff

# vLLM-Hook (cloned at $STEER_ROOT/vLLM-Hook)
cd ../vLLM-Hook
git apply grit/patches/vllm-hook_hook_llm.py.diff
git apply grit/patches/vllm-hook_steer_activation_worker.py.diff
```

`grit/scripts/toolbox_setup.sh` assumes these patches are already applied
before it does the `pip install --no-deps -e` of each package.

---

## What each patch does and why

### 1. `activation-steering_steering_vector.py.diff`

Two small bug fixes that surface when running on a modern torch/transformers
stack with a bf16 model:

1. **bf16 → numpy cast.** `batched_get_hiddens` calls
   `.squeeze().cpu().numpy()` on a bf16 tensor, which raises
   `TypeError: Got unsupported ScalarType BFloat16`. Adds `.float()` before
   the numpy conversion.

2. **JSON serialization of numpy scalars.** `SteeringVector.save` serializes
   `explained_variances` (a dict of `numpy.float32`) and the direction lists
   directly via `json.dump`, which raises
   `TypeError: Object of type float32 is not JSON serializable`. Casts both to
   native Python `float`/`list[float]`.

These are pure correctness fixes — the PCA direction extraction itself is
unchanged.

### 2. `vllm-hook_hook_llm.py.diff`

`HookLLM.__init__` constructs an `LLM` with a `worker_extension_cls` but never
triggers `install_hooks` on the engine. The general-plugin path
(`_hook_plugin.py`) normally does this lazily via a monkeypatched
`LLM.generate`; when constructing `LLM` directly (the offline/script path this
project uses), the hooks are registered on the worker mixin but never attached
to the model's decoder layers — so steering silently no-ops.

The patch calls `self.llm.collective_rpc("install_hooks")` after the engine is
created (with an engine-level fallback), so the worker's `install_hooks`
actually runs and attaches the forward hooks.

### 3. `vllm-hook_steer_activation_worker.py.diff`

The largest patch. The upstream worker targets the pre-v0.22 vLLM API, which
moved/broke in v0.22's V2 model runner:

- `model_runner.input_batch.req_ids` → no longer exists (the V2
  `GPUModelRunner` has no `input_batch`); replaced by `req_states`
  (a `RequestState` batch manager) with `index_to_req_id`.
- `model_runner.requests[req_id].sampling_params` → the `requests` dict is
  not reliably populated from inside the worker's forward hook on the V2
  runner, so per-request `extra_args["steer"]` lookups fail silently and
  steering no-ops.

The patch adds a **v0.22-safe global-config path**:

- Reads a steering config from the `VLLM_HOOK_STEER_CONFIG` env var at
  `install_hooks` time and applies it to the last token of every active batch
  row at the target layer — bypassing the per-request API entirely.
- Adds a `set_steering_active(active: bool)` RPC method so a caller can toggle
  steering on/off for A/B comparisons without reloading the engine.
- Keeps the legacy per-request path as a fallback when `model_runner.requests`
  *is* populated (older vLLM).

The steering math (`_apply_steer`, `add_vector` / `adjust_rs`) is unchanged.

---

## Verified on

- vLLM `0.22.1rc1.dev499+g470229c37.d20260613.rocm714`
- torch `2.13.0a0+rocm7.14.0a20260608`
- AMD Strix Halo, gfx1151

The v0.22 patches in particular are worth contributing upstream — the
`steer_hook_act` worker is otherwise unusable on any v0.22+ vLLM build.
