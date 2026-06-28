#!/usr/bin/env bash
# Run INSIDE the vllm toolbox to install the steering stack.
#
# *** INSTALL DISCIPLINE ***
# Every install here uses --no-deps. The toolbox ships an ABI-coupled stack
# (torch / vLLM / ROCm / aiter) compiled together; letting pip resolve deps
# would reinstall/upgrade torch or numpy and silently break that coupling.
# We only add pure-python leaf packages, and guard the known-vulnerable cores.
set -euo pipefail

STEER_ROOT="${STEER_ROOT:-/home/marcel/Work/steer}"

echo "============================================================"
echo "  GRIT STEERING — toolbox setup"
echo "  STEER_ROOT=$STEER_ROOT"
echo "============================================================"

# ROCm/HIP env for the toolbox (idempotent, runtime-only)
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Record the core versions BEFORE we touch anything, so we can detect drift.
BEFORE="$(python -c "
import importlib.metadata as m
for p in ('torch','numpy','vllm','transformers','scipy'):
    try: print(p, m.version(p))
    except Exception: print(p, 'MISSING')
")"
echo
echo "[pre-install] core versions:"
echo "$BEFORE"

# --- guard rails: refuse to proceed if a later step somehow tries to change them ---
core_check() {
    echo
    echo "[post-install] verifying core versions unchanged:"
    python -c "
import importlib.metadata as m, sys
before = '''$BEFORE'''
after = {}
for p in ('torch','numpy','vllm','transformers','scipy'):
    try: after[p] = m.version(p)
    except Exception: after[p] = 'MISSING'
drift = []
for line in before.strip().splitlines():
    p, v = line.split()
    if after.get(p) != v:
        drift.append(f'  {p}: {v} -> {after.get(p)}')
if drift:
    print('ABORTING — core package drift detected (ABI risk):')
    print('\n'.join(drift))
    sys.exit(1)
print('  OK — torch/numpy/vllm/transformers/scipy unchanged')
"
}

echo
echo "[1/3] Installing activation-steering (editable, --no-deps) ..."
# activation_steering is pure python; its deps (numpy, sklearn, torch, einops,
# transformers) are ALREADY in the toolbox. --no-deps keeps them pinned.
pip install --no-deps -e "$STEER_ROOT/activation-steering"

echo
echo "[2/3] Installing vLLM-Hook plugin (editable, --no-deps) ..."
pip install --no-deps -e "$STEER_ROOT/vLLM-Hook/vllm_hook_plugins"
# blake3/zstandard are small pure-python/C leaf deps of the hook; safe to add,
# and only if missing (no version bump of anything else).
pip install --no-deps --no-build-isolation blake3 2>/dev/null || \
    pip install --no-deps blake3 2>/dev/null || echo "  (blake3 already present or skipped)"
pip install --no-deps zstandard 2>/dev/null || echo "  (zstandard already present or skipped)"

echo
echo "[3/3] Smoke-testing imports ..."
python - <<'PY'
import sys
ok = True
for mod in ["vllm", "torch", "transformers", "numpy"]:
    try:
        m = __import__(mod)
        print(f"  OK   {mod}  {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        ok = False

# sklearn is a hard dep of the steering vector PCA step
try:
    import sklearn
    print(f"  OK   sklearn  {sklearn.__version__}")
except Exception as e:
    print(f"  FAIL sklearn: {e}  (pip install --no-deps scikit-learn if missing)")
    ok = False

# vLLM-Hook import is the real test — it pulls in every worker.
try:
    import vllm_hook_plugins
    vllm_hook_plugins.register_plugins()
    from vllm_hook_plugins import PluginRegistry
    workers = PluginRegistry.list_workers()
    print(f"  OK   vllm_hook_plugins  workers={workers}")
    assert "steer_hook_act" in workers, "steer_hook_act worker missing!"
    print("  OK   steer_hook_act worker registered")
except Exception as e:
    print(f"  FAIL vllm_hook_plugins: {e}")
    ok = False

try:
    from activation_steering import SteeringVector, SteeringDataset
    print("  OK   activation_steering (SteeringVector, SteeringDataset)")
except Exception as e:
    print(f"  FAIL activation_steering: {e}")
    ok = False

try:
    import torch
    print(f"  GPU visible: {torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch.cuda check failed: {e}")

sys.exit(0 if ok else 1)
PY

core_check

echo
echo "Setup complete. Next: train the grit vector."
echo "  python $STEER_ROOT/grit/scripts/train_grit_vector.py"
