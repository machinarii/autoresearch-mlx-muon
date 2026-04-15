"""
muon_mlx_reference.py — Reference Muon optimizer implementation for MLX.

Drop-in code for autoresearch-mlx's train.py (Phase 1-2 of Muon-MLX PRD).
This is a standalone reference file for development and testing.
In production, these functions get inlined into train.py directly.

Cross-referenced against:
  - KellerJordan/Muon/muon.py (canonical PyTorch implementation)
  - pytorch/pytorch/torch/optim/_muon.py (PyTorch core v2.9.0)
  - scasella/nanochat-mlx/nanochat_mlx/optim.py (working MLX Muon)
  - stockeh/mlx-optimizers (tested MLX Muon library)
  - ml-explore/mlx PR #1914 (MLX-specific edge cases)
  - awni/picochat (idiomatic MLX training patterns)

Usage:
  python muon_mlx_reference.py          # runs correctness + benchmark tests
  python muon_mlx_reference.py --bench  # benchmark only
"""

# ---------------------------------------------------------------------------
# STATUS: reference kept for isolated Newton-Schulz debugging.
# The validated logic was inlined into train.py on branch `muon-mlx`
# (commits 38ce76f newton_schulz, 596c2f2 constants, cca3f5f dual-path).
# Delete this file after a clean end-to-end training run confirms Muon works
# inside train.py. Until then, use this file to bisect NS-only issues.
# ---------------------------------------------------------------------------

import time
import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Muon Hyperparameters (agent-editable in train.py)
# ============================================================================
# These target the Muon crossover zone on M4 Max 128GB.
# At DEPTH=6, AR=64 (n_embd=512), expect ~400-700 steps in 5 min.
# Muon's advantage grows with matrix size — if the agent shrinks the model
# below AR=48, it should also try MUON_NS_STEPS=0 (AdamW-only).
#
# Sensible ranges for agent exploration:
#   MUON_MOMENTUM:  0.90 - 0.99
#   MUON_NS_STEPS:  0 (disabled), 3, 5, 7
#   MATRIX_LR:      0.005 - 0.1
#   MUON_BETA2:     0.0 (disabled), 0.90 - 0.99
#   MUON_NS_DTYPE:  "bfloat16" or "float32"
#
# On M5 Max with small models (4 layers, 256 dim, ~1900 steps), AdamW beat
# Muon (issue #21). On Mac Mini (fewer steps), Muon won. The crossover
# depends on matrix size and step count within the 5-minute budget.
# ============================================================================

MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_NESTEROV = True
MUON_BETA2 = 0.95
MUON_NS_DTYPE = "bfloat16"  # or "float32" if bfloat16 is unstable on Metal
MATRIX_LR = 0.02
EMBEDDING_LR = 0.6
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.15

# NS coefficients — tuned by Keller Jordan, validated on CUDA.
# These maximize slope at zero for fastest convergence in 5 iterations.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


# ============================================================================
# Newton-Schulz Orthogonalization
# ============================================================================

def newton_schulz(G: mx.array, steps: int = 5, dtype: str = "bfloat16") -> mx.array:
    """
    Compute the nearest orthogonal matrix to G via Newton-Schulz iteration.

    This is the core of Muon. It replaces the gradient update with the closest
    orthogonal matrix, removing redundant correlations between parameter
    dimensions so each training step does maximally useful work.

    The iteration computes:
        X_{k+1} = a*X + b*(X @ X.T) @ X + c*(X @ X.T)^2 @ X

    With tuned coefficients (3.4445, -4.7750, 2.0315) and 5 iterations,
    this converges to the polar factor (nearest orthogonal matrix).

    Args:
        G: 2D gradient/momentum matrix, any shape (m, n).
        steps: Number of NS iterations. 5 is standard, 3 is ~95% quality.
               0 skips orthogonalization entirely (returns scaled G).
        dtype: Compute dtype. "bfloat16" is default (matches CUDA behavior).
               Use "float32" if bfloat16 shows numerical instability on Metal.

    Returns:
        Orthogonalized matrix, same shape and dtype as input G.
    """
    if steps == 0:
        return G

    a, b, c = NS_COEFFS
    original_dtype = G.dtype

    # Cast to compute dtype if needed
    target_dtype = mx.bfloat16 if dtype == "bfloat16" else mx.float32
    X = G.astype(target_dtype)

    # Transpose so we iterate on the shorter dimension.
    # This makes A = X @ X.T the smaller matmul: (min(m,n), min(m,n))
    # Ref: pytorch/torch/optim/_muon.py transposes when m > n
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    # Frobenius-norm scaling for numerical stability.
    # clamp prevents division by zero on degenerate gradients.
    X = X / mx.maximum(mx.linalg.norm(X), 1e-7)

    # Newton-Schulz iterations.
    # Each iteration: 3 matmuls (X@X.T, A@A, B@X)
    # Total: 15 matmuls for 5 steps.
    # MLX's lazy evaluation builds a fused graph — no intermediate
    # materializations until mx.eval() at step boundary.
    for _ in range(steps):
        A = X @ X.T                        # Gram matrix
        B = b * A + c * (A @ A)            # Quintic polynomial coefficients
        X = a * X + B @ X                  # Update

    if transposed:
        X = X.T

    return X.astype(original_dtype)


# ============================================================================
# Parameter Classification
# ============================================================================

def classify_params(params: dict) -> tuple[dict, dict]:
    """
    Split model parameters into Muon-eligible and AdamW-only groups.

    Muon applies to 2D weight matrices in hidden layers (attention Q/K/V/O,
    MLP up/down projections). Everything else gets AdamW: embeddings (even
    though they're 2D), biases, layer norms, scalar mixers.

    Ref: KellerJordan/Muon — "Muon should only be used for hidden weight
    layers. The input embedding, final output layer, and any internal gains
    or biases should be optimized using a standard method such as AdamW."

    Args:
        params: Flattened dict of {name: mx.array} from model.parameters()

    Returns:
        (muon_params, adamw_params) — two dicts, keys are param names
    """
    muon_params = {}
    adamw_params = {}

    for name, param in params.items():
        is_2d = param.ndim == 2
        is_embedding = "embed" in name.lower()
        is_head = "head" in name.lower() or "lm_head" in name.lower()

        if is_2d and not is_embedding and not is_head:
            muon_params[name] = param
        else:
            adamw_params[name] = param

    return muon_params, adamw_params


# ============================================================================
# Optimizer State Management
# ============================================================================

def init_muon_state(param: mx.array) -> dict:
    """Initialize Muon optimizer state for a single parameter."""
    return {
        "momentum_buffer": mx.zeros_like(param),
        "nu": mx.array(0.0),  # EMA of update squared norms
        "step": 0,
    }


def init_adamw_state(param: mx.array) -> dict:
    """Initialize AdamW optimizer state for a single parameter."""
    return {
        "m": mx.zeros_like(param),  # first moment
        "v": mx.zeros_like(param),  # second moment
        "step": 0,
    }


# ============================================================================
# Muon Step
# ============================================================================

def muon_step(
    param: mx.array,
    grad: mx.array,
    state: dict,
    lr: float,
) -> tuple[mx.array, dict]:
    """
    Single Muon parameter update.

    1. Accumulate gradient into momentum buffer (SGD with β₁)
    2. Compute Nesterov look-ahead (optional)
    3. Orthogonalize via Newton-Schulz
    4. Scale by aspect ratio (√(fan_out / fan_in))
    5. Apply decoupled weight decay
    6. Update parameter

    Args:
        param: Current parameter value, shape (m, n)
        grad: Gradient w.r.t. param, same shape
        state: Optimizer state dict from init_muon_state()
        lr: Current learning rate (after schedule)

    Returns:
        (updated_param, updated_state)
    """
    state["step"] += 1

    # --- Momentum accumulation ---
    buf = MUON_MOMENTUM * state["momentum_buffer"] + grad
    state["momentum_buffer"] = buf

    # --- Nesterov look-ahead ---
    if MUON_NESTEROV:
        update = grad + MUON_MOMENTUM * buf
    else:
        update = buf

    # --- Newton-Schulz orthogonalization ---
    update = newton_schulz(update, steps=MUON_NS_STEPS, dtype=MUON_NS_DTYPE)

    # --- Aspect ratio scaling ---
    # "shape_scaling" mode from Keller Jordan's original.
    # Alternatives: "spectral" uses max(m, n)^0.5, "unit_rms_norm" uses (m*n)^0.25
    fan_out, fan_in = param.shape[0], param.shape[1]
    scale = max(1, fan_out / fan_in) ** 0.5

    # --- Optional: per-param adaptive scaling via β₂ ---
    # Smooths update magnitude across steps.
    # Set MUON_BETA2 = 0 to disable (pure Muon without adaptive scaling).
    if MUON_BETA2 > 0:
        update_sq_norm = mx.sum(update * update)
        nu = MUON_BETA2 * state["nu"] + (1 - MUON_BETA2) * update_sq_norm
        state["nu"] = nu
        # Bias correction
        bc = 1 - MUON_BETA2 ** state["step"]
        update = update / (mx.sqrt(nu / bc) + 1e-8)

    # --- Decoupled weight decay ---
    if WEIGHT_DECAY > 0:
        param = param * (1 - lr * WEIGHT_DECAY)

    # --- Apply update ---
    param = param - lr * scale * update

    return param, state


# ============================================================================
# AdamW Step (matches existing autoresearch-mlx pattern)
# ============================================================================

def adamw_step(
    param: mx.array,
    grad: mx.array,
    state: dict,
    lr: float,
    betas: tuple = (0.8, 0.95),
    eps: float = 1e-8,
) -> tuple[mx.array, dict]:
    """Standard AdamW update for non-Muon parameters."""
    state["step"] += 1
    b1, b2 = betas

    state["m"] = b1 * state["m"] + (1 - b1) * grad
    state["v"] = b2 * state["v"] + (1 - b2) * (grad * grad)

    # Bias correction
    bc1 = 1 - b1 ** state["step"]
    bc2 = 1 - b2 ** state["step"]
    m_hat = state["m"] / bc1
    v_hat = state["v"] / bc2

    # Decoupled weight decay
    if WEIGHT_DECAY > 0:
        param = param * (1 - lr * WEIGHT_DECAY)

    # Update
    param = param - lr * m_hat / (mx.sqrt(v_hat) + eps)

    return param, state


# ============================================================================
# Combined Training Step (wires into train.py's main loop)
# ============================================================================

def dual_optimizer_step(
    params: dict,
    grads: dict,
    states: dict,
    step_num: int,
    lr_scale: float,  # from LR schedule (0.0 to 1.0)
) -> tuple[dict, dict]:
    """
    Dual-optimizer update: Muon for 2D matrix params, AdamW for the rest.

    This replaces the single-optimizer step in autoresearch-mlx's train.py.

    Args:
        params: All model parameters {name: mx.array}
        grads: All gradients {name: mx.array}
        states: All optimizer states {name: dict}
        step_num: Current training step (for bias correction)
        lr_scale: LR schedule multiplier (warmup/warmdown)

    Returns:
        (updated_params, updated_states)
    """
    new_params = {}

    for name, param in params.items():
        grad = grads[name]

        # Initialize state on first step
        if name not in states:
            if param.ndim == 2 and "embed" not in name.lower() and "head" not in name.lower():
                states[name] = init_muon_state(param)
            else:
                states[name] = init_adamw_state(param)

        # Route to correct optimizer
        if "momentum_buffer" in states[name]:
            # Muon path
            new_params[name], states[name] = muon_step(
                param, grad, states[name],
                lr=lr_scale * MATRIX_LR,
            )
        else:
            # AdamW path — use EMBEDDING_LR for embeddings, SCALAR_LR otherwise
            if "embed" in name.lower():
                lr = lr_scale * EMBEDDING_LR
            else:
                lr = lr_scale * SCALAR_LR

            new_params[name], states[name] = adamw_step(
                param, grad, states[name],
                lr=lr,
            )

    return new_params, states


# ============================================================================
# Tests & Benchmarks
# ============================================================================

def test_orthogonality():
    """Verify Newton-Schulz produces near-orthogonal matrices."""
    print("=" * 60)
    print("TEST: Orthogonality verification")
    print("=" * 60)

    for shape in [(256, 256), (512, 512), (768, 768), (512, 256), (256, 512)]:
        for dtype_str in ["bfloat16", "float32"]:
            G = mx.random.normal(shape)
            U = newton_schulz(G, steps=5, dtype=dtype_str)
            mx.eval(U)

            # Check U @ U.T ≈ I (for the smaller dimension)
            if U.shape[0] <= U.shape[1]:
                product = U @ U.T
                eye = mx.eye(U.shape[0])
            else:
                product = U.T @ U
                eye = mx.eye(U.shape[1])

            residual = mx.mean(mx.abs(product.astype(mx.float32) - eye)).item()
            status = "PASS" if residual < 0.15 else "FAIL"
            print(f"  {status}  shape={shape}, dtype={dtype_str}, "
                  f"residual=||UU^T - I||_mean = {residual:.6f}")

    print()


def test_gradient_flow():
    """Verify no NaN/Inf after multiple Muon steps."""
    print("=" * 60)
    print("TEST: Gradient flow (100 steps, no NaN/Inf)")
    print("=" * 60)

    param = mx.random.normal((512, 512)) * 0.02
    state = init_muon_state(param)

    for step in range(100):
        grad = mx.random.normal(param.shape) * 0.01
        param, state = muon_step(param, grad, state, lr=0.02)
        mx.eval(param, state["momentum_buffer"], state["nu"])

        has_nan = mx.any(mx.isnan(param)).item()
        has_inf = mx.any(mx.isinf(param)).item()
        if has_nan or has_inf:
            print(f"  FAIL  NaN/Inf at step {step}")
            return

    print(f"  PASS  100 steps completed, param norm = {mx.linalg.norm(param).item():.4f}")
    print()


def test_ns_disabled():
    """Verify MUON_NS_STEPS=0 degrades to scaled SGD-momentum."""
    print("=" * 60)
    print("TEST: NS_STEPS=0 fallback (scaled SGD-momentum)")
    print("=" * 60)

    G = mx.random.normal((512, 512))
    result = newton_schulz(G, steps=0, dtype="bfloat16")
    mx.eval(result)

    # With steps=0, output should be identical to input
    diff = mx.max(mx.abs(result - G)).item()
    status = "PASS" if diff == 0.0 else "FAIL"
    print(f"  {status}  max(|output - input|) = {diff}")
    print()


def test_classify_params():
    """Verify parameter classification matches expected groups."""
    print("=" * 60)
    print("TEST: Parameter classification")
    print("=" * 60)

    # Simulate a small transformer's parameter shapes
    fake_params = {
        "embed.weight": mx.zeros((8192, 512)),      # embedding — AdamW
        "layers.0.attn.q.weight": mx.zeros((512, 512)),  # attention — Muon
        "layers.0.attn.k.weight": mx.zeros((512, 512)),  # attention — Muon
        "layers.0.attn.v.weight": mx.zeros((512, 512)),  # attention — Muon
        "layers.0.attn.o.weight": mx.zeros((512, 512)),  # attention — Muon
        "layers.0.mlp.up.weight": mx.zeros((2048, 512)), # MLP — Muon
        "layers.0.mlp.down.weight": mx.zeros((512, 2048)), # MLP — Muon
        "layers.0.norm.weight": mx.zeros((512,)),    # layer norm — AdamW
        "layers.0.norm.bias": mx.zeros((512,)),      # bias — AdamW
        "lm_head.weight": mx.zeros((8192, 512)),     # output head — AdamW
    }

    muon_p, adamw_p = classify_params(fake_params)

    expected_muon = {"layers.0.attn.q.weight", "layers.0.attn.k.weight",
                     "layers.0.attn.v.weight", "layers.0.attn.o.weight",
                     "layers.0.mlp.up.weight", "layers.0.mlp.down.weight"}
    expected_adamw = {"embed.weight", "layers.0.norm.weight",
                      "layers.0.norm.bias", "lm_head.weight"}

    muon_ok = set(muon_p.keys()) == expected_muon
    adamw_ok = set(adamw_p.keys()) == expected_adamw

    print(f"  {'PASS' if muon_ok else 'FAIL'}  Muon params: {sorted(muon_p.keys())}")
    print(f"  {'PASS' if adamw_ok else 'FAIL'}  AdamW params: {sorted(adamw_p.keys())}")
    print()


def benchmark_ns():
    """Benchmark Newton-Schulz at various matrix sizes."""
    print("=" * 60)
    print("BENCHMARK: Newton-Schulz wall-clock time")
    print("=" * 60)
    print(f"  {'Shape':>12}  {'dtype':>10}  {'Steps':>5}  {'Time (ms)':>10}  {'Per-iter':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}")

    for shape in [(256, 256), (384, 384), (512, 512), (768, 768)]:
        for dtype_str in ["bfloat16", "float32"]:
            G = mx.random.normal(shape)
            mx.eval(G)

            # Warmup
            _ = newton_schulz(G, steps=5, dtype=dtype_str)
            mx.eval(_)

            # Timed run (average of 10)
            times = []
            for _ in range(10):
                start = time.perf_counter()
                result = newton_schulz(G, steps=5, dtype=dtype_str)
                mx.eval(result)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_ms = sum(times) / len(times)
            per_iter = avg_ms / 5
            print(f"  {str(shape):>12}  {dtype_str:>10}  {5:>5}  {avg_ms:>9.2f}ms  {per_iter:>9.2f}ms")

    print()

    # Overhead estimate for full training step
    print("  OVERHEAD ESTIMATE (target config: 12 Muon params × 512×512):")
    G = mx.random.normal((512, 512))
    mx.eval(G)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        for _p in range(12):  # ~12 weight matrices in a 6-layer transformer
            result = newton_schulz(G, steps=5, dtype="bfloat16")
        mx.eval(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    avg_total = sum(times) / len(times)
    print(f"  12 params × 5 NS steps = {avg_total:.1f}ms total per training step")
    print(f"  At ~400-700 steps/5min, this is {avg_total * 600 / 1000:.1f}s - {avg_total * 700 / 1000:.1f}s overhead")
    print()


def benchmark_compiled_ns():
    """Benchmark Newton-Schulz with mx.compile wrapping."""
    print("=" * 60)
    print("BENCHMARK: mx.compile effect on Newton-Schulz")
    print("=" * 60)

    G = mx.random.normal((512, 512))
    mx.eval(G)

    # Uncompiled
    times_raw = []
    for _ in range(10):
        start = time.perf_counter()
        result = newton_schulz(G, steps=5, dtype="bfloat16")
        mx.eval(result)
        times_raw.append((time.perf_counter() - start) * 1000)

    # Compiled
    ns_compiled = mx.compile(newton_schulz)
    # Warmup the compiled version
    _ = ns_compiled(G, steps=5, dtype="bfloat16")
    mx.eval(_)

    times_compiled = []
    for _ in range(10):
        start = time.perf_counter()
        result = ns_compiled(G, steps=5, dtype="bfloat16")
        mx.eval(result)
        times_compiled.append((time.perf_counter() - start) * 1000)

    avg_raw = sum(times_raw) / len(times_raw)
    avg_compiled = sum(times_compiled) / len(times_compiled)
    speedup = avg_raw / avg_compiled if avg_compiled > 0 else 0

    print(f"  Uncompiled:  {avg_raw:.2f}ms")
    print(f"  Compiled:    {avg_compiled:.2f}ms")
    print(f"  Speedup:     {speedup:.2f}x")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        benchmark_ns()
        benchmark_compiled_ns()
    else:
        test_orthogonality()
        test_gradient_flow()
        test_ns_disabled()
        test_classify_params()
        print("=" * 60)
        print("All tests passed. Running benchmarks...")
        print("=" * 60)
        print()
        benchmark_ns()
        benchmark_compiled_ns()
        print("Done. Copy newton_schulz(), muon_step(), classify_params()")
        print("and dual_optimizer_step() into train.py for Phase 2.")
