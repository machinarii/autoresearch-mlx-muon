# PRD: Muon Optimizer for MLX

**Project:** `muon-mlx` — Native Apple Silicon Muon optimizer for autoresearch-mlx  
**Author:** Jin  
**Date:** April 2026  
**Status:** Draft  
**Target repo:** [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)  
**Reference impl:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch) `train.py` (PyTorch/CUDA)

---

## 1. Problem Statement

The autoresearch-mlx fork currently uses AdamW as its sole optimizer. The upstream autoresearch repo uses a dual-optimizer strategy: **Muon for 2D weight matrices** (attention projections, MLP layers) and **AdamW for everything else** (embeddings, biases, layer norms, scalar mixing parameters). Muon delivers ~35% faster convergence per step compared to AdamW on transformer weight matrices by orthogonalizing gradient updates via Newton-Schulz iteration.

However, recent empirical results from the autoresearch-mlx community (issue #21: M5 Max 128GB, 25 experiments) reveal a critical nuance: **Muon's advantage is hardware- and model-size-dependent on Apple Silicon.** On the M5 Max, AdamW won outright — the agent ran 25 experiments and never adopted Muon. On the Mac Mini (slower hardware), Muon won. The deciding factor is step count: when the hardware is fast enough to push ~1900 steps in 5 minutes with a small model (4 layers, 256 dim), the "more steps with AdamW" strategy beats the "fewer but better steps with Muon" strategy. Muon's advantage grows with matrix size, where gradient correlations between rows become significant enough that orthogonalization provides meaningful per-step gains.

The M4 Max 128GB sits in an interesting position: slower than the M5 Max (fewer steps in the same budget) but with identical memory headroom. More importantly, the 128GB allows pushing to larger model configs (ASPECT_RATIO 64-96, depth 6-8) where Muon's per-step advantage should dominate — a regime no Apple Silicon user has publicly validated yet.

This PRD specifies a native MLX implementation of Muon that integrates into autoresearch-mlx's existing `train.py`, expanding the optimizer search space so the autonomous agent can discover the hardware-specific crossover point between AdamW and Muon on the M4 Max.

---

## 2. Goals

**Primary goal:** Implement Muon in pure MLX so that autoresearch-mlx's agent loop can explore a dual-optimizer strategy alongside architecture scaling, enabling the agent to discover the hardware-specific crossover point where Muon outperforms AdamW on the M4 Max 128GB.

**Secondary goals:**

- Keep the implementation self-contained within `train.py` (no separate optimizer module), matching autoresearch's single-file philosophy.
- Maintain the dual-optimizer pattern: Muon for 2D matrix parameters, AdamW for 1D/scalar parameters.
- Expose the same hyperparameter surface the agent can explore: `momentum`, `ns_steps`, `beta2`, `matrix_lr`, `nesterov`.
- **Default to a model config large enough for Muon to matter** (ASPECT_RATIO ≥ 64, depth ≥ 6), leveraging the M4 Max's 128GB memory advantage over smaller Apple Silicon machines.
- Target M4 Max (128GB unified memory) as the primary development platform, with correctness on M1/M2/M3 as a constraint.
- Keep per-step overhead of Newton-Schulz iteration under 5% of total step time for the target model configuration.

**Non-goals:**

- Proving Muon universally beats AdamW on Apple Silicon (issue #21 shows it doesn't for small models on fast hardware — that's expected behavior, not a failure).
- Distributed/multi-device Muon (no `mx.distributed` integration).
- Newton-Muon or Turbo-Muon variants (these are possible follow-ups but out of scope).
- Custom Metal kernels (use MLX's existing matmul and lazy evaluation).
- Compatibility with PyTorch or the MPS-based autoresearch-macos fork.

---

## 3. Background: How Muon Works

Muon (MomentUm Orthogonalized by Newton-schulz) is a two-phase optimizer:

**Phase 1 — Momentum accumulation.** Standard SGD with Nesterov momentum computes a running average of gradients. For parameter `W` with gradient `G`:

```
momentum_buffer = β₁ * momentum_buffer + G
update = G + β₁ * momentum_buffer    # Nesterov look-ahead
```

**Phase 2 — Newton-Schulz orthogonalization.** The momentum-accumulated update (a 2D matrix) is replaced with its nearest orthogonal matrix via iterative Newton-Schulz:

```
X₀ = update / ‖update‖_F          # Frobenius-norm scaling
For k = 0..ns_steps-1:
    X_{k+1} = a·Xₖ + b·Xₖ·Xₖᵀ·Xₖ + c·Xₖ·(Xₖᵀ·Xₖ)²
```

With tuned coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` and 5 iterations, this converges to the polar factor (nearest orthogonal matrix) of the update.

**Phase 3 — Scaling and application.** The orthogonalized update is scaled by `√(fan_out / fan_in)` (or `√max(fan_out, fan_in)` in spectral mode) and applied with the matrix learning rate:

```
W = W - lr * scale_factor * orthogonalized_update
```

**Why it helps:** Orthogonal updates remove harmful correlations between parameter dimensions (e.g., Q and K in attention are coupled through their dot product). Each step moves the weights in a maximally informative direction, extracting more learning per forward-backward pass.

**Computational cost:** Each NS iteration requires 3 matrix multiplications (matmul). For 5 iterations on a weight matrix of shape `(d, d)`, that's 15 matmuls per parameter group per step — cheap relative to the forward/backward pass for transformer-scale matrices.

---

## 3.1 The Model Size Crossover: When Muon Wins vs Loses on Apple Silicon

Community results from autoresearch-mlx (issue #21, April 2026) establish that Muon's advantage is not universal on Apple Silicon. The key variable is the ratio of step count to matrix size within the 5-minute budget.

**Where AdamW wins (small model, high step count):**

The M5 Max 128GB user ran 25 experiments with the default small config (4 layers, n_embd=256). The winning recipe achieved ~1900 steps in 5 minutes with val_bpb 1.235. Weight matrices at 256×256 are small enough that gradient correlations between rows are minimal — orthogonalization adds overhead without meaningful learning benefit. The agent correctly never adopted Muon. SiLU activation was also catastrophic (1.301 → 1.768), and the dominant optimization lever was gradual LR reduction (0.04 → 0.005).

**Where Muon wins (larger model, lower step count):**

The Mac Mini (slower hardware, fewer steps per budget) found Muon beneficial. The upstream H100 results (SkyPilot: ASPECT_RATIO=96, ~1060 steps in 5 minutes) also show Muon as a late-stage improvement after architecture scale was optimized. At 512×512 or 768×768, weight matrices have enough internal structure that orthogonalization removes real redundancy in the gradient.

**M4 Max 128GB positioning:**

| Config | Est. steps/5min | Matrix size | Muon prediction |
|---|---|---|---|
| 4 layers, 256 dim | ~1200-1500 | 256×256 | Likely ignored (too small) |
| 4-6 layers, 384 dim | ~600-1000 | 384×384 | Crossover zone |
| **6 layers, 512 dim** | **~400-700** | **512×512** | **Likely beneficial** |
| 8 layers, 768 dim | ~200-400 | 768×768 | Strong Muon advantage |

The 128GB memory ceiling means the M4 Max can comfortably hold configs up to ASPECT_RATIO=96 (768 dim, 8 layers) without OOM risk — a regime where the M5 Max user's depth-6 crash wouldn't occur. **The default config for Muon validation should therefore target the crossover zone or above.**

**Recommended default config for Muon evaluation:**

```python
DEPTH = 6
ASPECT_RATIO = 64          # n_embd = 512
TOTAL_BATCH_SIZE = 2**16   # 65K tokens/step
MATRIX_LR = 0.02           # Muon LR
MUON_NS_STEPS = 5          # Agent can toggle to 0 for AdamW-only comparison
```

This config targets ~400-700 steps in 5 minutes on the M4 Max. The agent should be free to scale up or down from here — the key insight is that the starting point determines whether Muon gets a fair trial.

---

## 4. Technical Specification

### 4.1 Core Implementation: `newton_schulz_mlx()`

The Newton-Schulz orthogonalization function, implemented in pure MLX. The implementation should cross-reference three existing MLX Muon ports for patterns and edge cases: `scasella/nanochat-mlx` `optim.py` (working in-training-loop integration), `stockeh/mlx-optimizers` (tested library with multiple NS backends), and the discussion in `ml-explore/mlx` PR #1914 (tensor dimensionality handling). For idiomatic MLX training patterns — especially `mx.compile` wrapping and `mx.eval` placement — reference `awni/picochat` by MLX core team member Awni Hannun.

```python
import mlx.core as mx

def newton_schulz_mlx(G: mx.array, steps: int = 5) -> mx.array:
    """
    Compute the nearest orthogonal matrix to G via Newton-Schulz iteration.
    
    Args:
        G: 2D gradient/momentum matrix, shape (m, n) where m >= n.
            If m < n, transpose internally, compute, transpose back.
        steps: Number of NS iterations. 5 is standard.
    
    Returns:
        Orthogonalized matrix, same shape as G.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True
    
    # Frobenius-norm scaling for numerical stability
    X = G / (mx.linalg.norm(G) + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T          # (m, m)
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    
    if transposed:
        X = X.T
    
    return X
```

**MLX-specific considerations:**

- **Lazy evaluation:** MLX builds computation graphs lazily. The 5-iteration loop creates a graph of ~15 matmuls that fuses efficiently when `mx.eval()` is called at step boundary. No explicit kernel fusion needed.
- **`mx.compile` compatibility:** The function uses only standard MLX ops (`@` matmul, `mx.linalg.norm`, scalar multiply, addition). All are compatible with `mx.compile` for graph-level optimization.
- **Dtype:** Use the same dtype as the model (typically `mx.bfloat16` or `mx.float16`). The NS coefficients are float64 literals but MLX will downcast at the scalar-array multiply boundary. The iteration is empirically stable in bfloat16 on GPU (confirmed on CUDA; needs validation on Metal — see Section 6).
- **Tall-skinny handling:** If `m < n`, transpose so the iteration runs on the (n, m) shape and transpose back. This ensures `A = X @ X.T` is the smaller `(min(m,n), min(m,n))` matmul.

### 4.2 Muon Optimizer Class

Implemented as a stateful step function within `train.py`, not as an `mlx.optimizers.Optimizer` subclass — matching autoresearch's flat single-file pattern:

```python
# ---- Hyperparameters (agent-editable) ----
# Architecture defaults targeting the Muon crossover zone on M4 Max 128GB.
# At DEPTH=6, AR=64 (n_embd=512), expect ~400-700 steps in 5 min.
# Muon's advantage grows with matrix size — if the agent shrinks the model
# below AR=48, it should also try MUON_NS_STEPS=0 (AdamW-only).
MUON_MOMENTUM = 0.95          # β₁ for SGD momentum
MUON_NS_STEPS = 5             # Newton-Schulz iterations (0 = disable Muon, use AdamW-only)
MUON_NESTEROV = True          # Nesterov look-ahead
MUON_BETA2 = 0.95             # EMA for per-param adaptive LR (optional)
MATRIX_LR = 0.02              # Learning rate for Muon-optimized params
EMBEDDING_LR = 0.6            # AdamW LR for embeddings
SCALAR_LR = 0.5               # AdamW LR for scalars/biases

# Weight decay applied uniformly
WEIGHT_DECAY = 0.15           # M5 Max issue #21 found 0.15 optimal
```

**Parameter classification logic:**

```python
def classify_params(model):
    """
    Split model parameters into two groups:
      - matrix_params: 2D weight matrices (attention Q/K/V/O, MLP up/down)
                       → optimized by Muon
      - scalar_params: embeddings, biases, layer norms, scalar mixers
                       → optimized by AdamW
    """
    matrix_params = {}
    scalar_params = {}
    
    for name, param in model.parameters().items():
        if param.ndim == 2 and 'embed' not in name:
            matrix_params[name] = param
        else:
            scalar_params[name] = param
    
    return matrix_params, scalar_params
```

**Muon state per parameter:**

| State variable | Shape | Purpose |
|---|---|---|
| `momentum_buffer` | same as param | Running momentum (SGD with β₁) |
| `nu` (optional) | scalar | EMA of update squared norms for adaptive scaling (β₂) |

**AdamW state per parameter (unchanged from current MLX fork):**

| State variable | Shape | Purpose |
|---|---|---|
| `m` | same as param | First moment |
| `v` | same as param | Second moment |

### 4.3 Training Loop Integration

The step function replaces the current single-optimizer `adamw_step` with a dual-path update:

```python
def muon_step(param, grad, state, lr, step_num):
    """Single Muon parameter update."""
    # Momentum
    buf = state['momentum_buffer']
    buf = MUON_MOMENTUM * buf + grad
    state['momentum_buffer'] = buf
    
    # Nesterov look-ahead
    if MUON_NESTEROV:
        update = grad + MUON_MOMENTUM * buf
    else:
        update = buf
    
    # Newton-Schulz orthogonalization
    update = newton_schulz_mlx(update, steps=MUON_NS_STEPS)
    
    # Aspect ratio scaling
    scale = max(1, param.shape[0] / param.shape[1]) ** 0.5
    
    # Optional: per-param adaptive scaling via β₂
    if MUON_BETA2 > 0:
        nu = state.get('nu', 0.0)
        update_norm_sq = mx.sum(update * update).item()
        nu = MUON_BETA2 * nu + (1 - MUON_BETA2) * update_norm_sq
        state['nu'] = nu
        # Scale by inverse RMS of recent updates
        update = update / (mx.sqrt(mx.array(nu)) + 1e-8)
    
    # Weight decay (decoupled)
    if WEIGHT_DECAY > 0:
        param = param * (1 - lr * WEIGHT_DECAY)
    
    # Apply update
    param = param - lr * scale * update
    return param, state


def training_step(model, batch, optimizer_states, step_num, lr_schedule):
    """
    Combined forward + backward + dual optimizer update.
    """
    loss, grads = loss_and_grad_fn(model, batch)
    
    matrix_lr = lr_schedule(step_num) * MATRIX_LR
    adamw_lr = lr_schedule(step_num)
    
    new_params = {}
    for name, param in model.parameters().items():
        grad = grads[name]
        
        if param.ndim == 2 and 'embed' not in name:
            # Muon path
            param, optimizer_states[name] = muon_step(
                param, grad, optimizer_states[name],
                matrix_lr, step_num
            )
        else:
            # AdamW path (existing implementation)
            param, optimizer_states[name] = adamw_step(
                param, grad, optimizer_states[name],
                adamw_lr, step_num
            )
        
        new_params[name] = param
    
    model.update(new_params)
    mx.eval(model.parameters(), optimizer_states)  # Force eval at step boundary
    
    return loss
```

### 4.4 Memory Budget

Muon adds one momentum buffer per 2D parameter (same memory as AdamW's first moment). Since we're replacing AdamW's two buffers (`m`, `v`) with Muon's one buffer (`momentum_buffer`) plus an optional scalar `nu`, the Muon path actually uses **less memory** than AdamW per matrix parameter. Net memory change: **negative** (saves ~1 buffer per matrix param).

For the default model config (DEPTH=8, model_dim ~384):

| Component | AdamW-only (current) | Muon + AdamW (proposed) |
|---|---|---|
| Matrix param states | 2 buffers each | 1 buffer each (momentum only) |
| Scalar param states | 2 buffers each | 2 buffers each (unchanged) |
| NS temporaries | — | 2 matrices during iteration (freed after) |
| **Net change** | baseline | **~30-40% less optimizer memory** |

On the M4 Max with 128GB unified memory, optimizer state memory is not a constraint at any plausible model size within the 5-minute training budget.

### 4.5 Compilation Strategy

Wrap the Newton-Schulz function with `mx.compile` to fuse the iteration graph:

```python
newton_schulz_compiled = mx.compile(newton_schulz_mlx)
```

MLX's `mx.compile` performs graph-level fusion, eliminating intermediate materializations between the matmuls in the NS loop. This is the MLX equivalent of what `torch.compile` does for the CUDA path — but unlike MPS, MLX's compile actually works on Apple Silicon.

---

## 5. Acceptance Criteria

### 5.1 Correctness

- **Numerical parity:** Given the same initial weights, learning rate, and random seed, the Muon-optimized parameters should produce orthogonal-ish updates (verify that `U @ U.T ≈ I` after NS iteration, within bfloat16 tolerance).
- **Gradient flow:** No NaN/Inf in any parameter after 1000 steps on the default model config.
- **Parameter classification:** Exactly the same parameters get Muon vs AdamW as in the upstream CUDA version. Embeddings, biases, and norms must NOT go through NS iteration.

### 5.2 Performance

- **val_bpb improvement at target scale:** On M4 Max 128GB with the target config (DEPTH=6, AR=64, batch 2^16), the Muon+AdamW config should achieve lower val_bpb than AdamW-only after a 5-minute training run. Target: ≥5% improvement at this model scale.
- **Correct behavior at small scale:** At the default small config (DEPTH=4, AR=32, n_embd=256), it is acceptable — and expected — for the agent to discover that `MUON_NS_STEPS=0` (AdamW-only) produces better val_bpb. This validates that the implementation doesn't force Muon where it doesn't help.
- **Step overhead:** Newton-Schulz iteration should add <5% wall-clock time per training step at the target scale (measured as the delta between a Muon step and an AdamW step for the same 512×512 parameter).
- **Experiment throughput:** The ~7 minute experiment cycle (5 min training + compile + eval) should not increase by more than 30 seconds total.

### 5.3 Integration

- **Single-file:** Everything lives in `train.py`. No new files, no new dependencies.
- **Agent-safe:** All Muon hyperparameters are top-level constants the agent can modify. The agent can disable Muon entirely by setting `MUON_NS_STEPS = 0` (skip orthogonalization, fallback to scaled SGD-momentum).
- **Backward compatible:** Running the new `train.py` with `MUON_NS_STEPS = 0` reproduces the current AdamW-only behavior within numerical tolerance.

---

## 6. Risks and Mitigations

### 6.1 bfloat16 Stability on Metal

**Risk:** The NS iteration involves repeated matmuls that could amplify numerical errors differently on Metal's bfloat16 pipeline vs CUDA's tensor cores. The tuned coefficients `(3.4445, -4.7750, 2.0315)` were optimized for CUDA numerical behavior.

**Mitigation:** 
- Run NS iteration in float32 if bfloat16 shows divergence, casting back to bfloat16 for the final update. MLX's unified memory makes this cast essentially free (no copy).
- Add a `MUON_NS_DTYPE` hyperparameter (`"float32"` or `"bfloat16"`) so the agent can discover which is better empirically.
- Validate orthogonality residual `‖X·Xᵀ - I‖_F` across 100 steps on both dtypes.

### 6.2 Matmul Throughput on Apple Silicon

**Risk:** Newton-Schulz adds ~15 matmuls per parameter group per step. Apple Silicon's matmul throughput is ~6-7x slower than H100 per-op. If NS overhead exceeds 5% of step time, it cuts into the fixed-time budget. At the target config (512×512 matrices, ~400-700 steps), this is 15 × N_params × 500 = thousands of extra matmuls per experiment.

**Mitigation:**
- MLX's lazy evaluation + `mx.compile` should fuse the 15 matmuls per param into a single kernel launch. Benchmark this explicitly.
- At 512×512, each NS matmul is ~0.27 GFLOPs — still small relative to the forward/backward pass matmuls operating on batch-sized tensors. The NS overhead is dominated by kernel launch, which fusion eliminates.
- If overhead is too high, reduce `MUON_NS_STEPS` to 3 (the NS iteration converges rapidly; 3 steps gives ~95% of the orthogonality quality of 5 steps).
- Issue #21's data gives a natural comparison: if per-step time at the target scale with Muon exceeds the per-step time without Muon by >10%, the matrices are too small or fusion isn't working.

### 6.3 Model Size Regime Mismatch

**Risk:** If the default model config is too small (DEPTH=4, AR=32), Muon will be correctly rejected by the agent in the first few experiments, and the overnight run will never explore Muon at larger scales. The implementation would be validated as "working" but never tested in the regime where it matters. This is what likely happened on the M5 Max in issue #21 — the agent started small, found AdamW winning, and never scaled up enough for Muon to cross over.

**Mitigation:**
- Set the default config to DEPTH=6, AR=64 (n_embd=512), targeting the crossover zone where Muon has a realistic chance of winning.
- Add a comment in `program.md` explicitly telling the agent: "If you scale the model below AR=48, also test MUON_NS_STEPS=0 vs 5 — Muon may not help at small matrix sizes."
- Run the Phase 5 overnight validation at two scales: the small default (to verify AdamW still wins there) and the target config (to verify Muon can win).
- Include the issue #21 M5 Max results in `program.md` as empirical context so the agent understands the tradeoff.

### 6.4 Agent Interaction

**Risk:** The autoresearch agent might not know how to tune Muon-specific hyperparameters, or might disable Muon inadvertently by setting a bad learning rate. Conversely, the agent might blindly keep Muon enabled at small scales where it hurts.

**Mitigation:**
- Add a comment block above the Muon hyperparameters explaining what each one does, sensible ranges, and the model-size dependency.
- Set defaults that target the Muon-favorable regime, so the baseline gives Muon a fair shot.
- The agent can always compare Muon on/off by toggling `MUON_NS_STEPS` between 5 and 0.
- Include issue #21's finding in `program.md`: "On the M5 Max with small models, AdamW outperformed Muon. On the Mac Mini with the same codebase, Muon won. Model size and step count determine which optimizer is better."

---

## 7. Implementation Plan

### Phase 1: Newton-Schulz Core (1-2 hours)

- Implement `newton_schulz_mlx()` as a standalone function.
- Unit test: verify orthogonality on random matrices of various shapes.
- Benchmark: measure wall-clock time for 5 NS iterations on (256, 256), (384, 384), (512, 512), (768, 768) matrices in bfloat16 and float32. The 512×512 case is the primary target; 256×256 is the "too small for Muon" reference; 768×768 is the aspirational upper bound.

### Phase 2: Optimizer Integration (2-3 hours)

- Implement `muon_step()` and parameter classification.
- Wire into the existing training loop as a dual-path update.
- Verify gradient flow: train 100 steps, check no NaN/Inf, check loss is decreasing.

### Phase 3: Validation (2-3 hours)

- Run a full 5-minute training experiment with Muon+AdamW.
- Compare val_bpb against the current AdamW-only baseline.
- Profile per-step timing to confirm NS overhead is <5%.
- Test bfloat16 vs float32 NS iteration stability.

### Phase 4: Agent-Readiness (1 hour)

- Add hyperparameter comment block.
- Test that `MUON_NS_STEPS = 0` reproduces AdamW-only behavior.
- Update `program.md` to inform the agent about the new optimizer and its tunable parameters.
- Commit as a single clean diff to `train.py`.

### Phase 5: autoresearch-mlx Fork Integration & Overnight Validation (4-8 hours)

This phase takes the validated Muon implementation from Phases 1-4 and lands it into a working fork of autoresearch-mlx, then validates it under real autonomous research conditions.

**5a. Fork Sync and Dual-Scale Baseline (1-2 hours)**

- Fork `trevin-creator/autoresearch-mlx` (or work from existing clone).
- Sync with upstream `main` to pick up any recent changes — the README mentions a Muon variant was explored in the Mac Mini run but not published; check if any traces landed. Also check the `autoresearch/m5max-apr5` branch referenced in issue #21 for the M5 Max results.
- Run two baselines on M4 Max 128GB to establish AdamW-only performance at both scales:
  - **Small scale:** Default config (DEPTH=4, AR=32, n_embd=256, batch 2^15). Record step count, per-step timing, val_bpb, peak memory via `mx.metal.get_peak_memory()`. This is the regime where Muon should lose (per issue #21).
  - **Target scale:** DEPTH=6, AR=64, n_embd=512, batch 2^16. Same measurements. This is the regime where Muon should win.
- Commit both baselines to `results.tsv` with clear labels.

**5b. Merge Muon into `train.py` (1-2 hours)**

- Port the validated `newton_schulz_mlx()`, `muon_step()`, and parameter classification logic into the fork's `train.py`.
- Cross-reference against `scasella/nanochat-mlx` `optim.py` and `stockeh/mlx-optimizers` Muon class for any MLX-specific patterns missed in Phase 2 (e.g., how they handle `mx.eval` timing, memory cleanup between steps, state tree structure for `mlx.optimizers`-compatible state dicts).
- Cross-reference against `awni/picochat` for MLX training loop patterns — picochat is by Awni Hannun (MLX core team), so its training patterns represent idiomatic MLX usage (e.g., how `mx.compile` wraps the step function, gradient accumulation, eval placement).
- Preserve the existing AdamW code path so `MUON_NS_STEPS = 0` reverts cleanly.
- Verify the fork still passes a quick smoke test (5-minute training run completes, no crashes).

**5c. Update `program.md` (30 min)**

- Add a section to `program.md` informing the agent about the dual-optimizer setup.
- Include the issue #21 empirical finding as context: "On the M5 Max with a small model (4 layers, 256 dim, ~1900 steps), AdamW outperformed Muon. On the Mac Mini (slower, fewer steps), Muon won. Muon's advantage grows with matrix size — if you scale below AR=48, also test MUON_NS_STEPS=0."
- Document which hyperparameters are Muon-specific and their sensible search ranges:
  - `MUON_MOMENTUM`: 0.9 - 0.99 (default 0.95)
  - `MUON_NS_STEPS`: 0 (disabled), 3, 5, 7 (default 5)
  - `MATRIX_LR`: 0.005 - 0.1 (default 0.02)
  - `MUON_BETA2`: 0.0 (disabled), 0.9 - 0.99 (default 0.95)
  - `MUON_NS_DTYPE`: "bfloat16" or "float32"
- Include the M5 Max's winning non-Muon findings as hints the agent can try: weight_decay=0.15, warmup=0.1, warmdown=0.4, ReLU² (not SiLU), VE on all layers, no x0 connection.

**5d. Overnight Autonomous Run (overnight, ~8-10 hours)**

- Launch the autoresearch loop with Claude Code pointed at the updated `program.md`.
- **Start at the target scale** (DEPTH=6, AR=64, batch 2^16) with Muon enabled — NOT the small default. The agent can scale down if it wants, but starting large gives Muon a fair trial.
- Target: ~70-80 experiments overnight on M4 Max.
- First experiment should be the Muon-enabled baseline at target scale vs the AdamW-only baseline at the same scale, to isolate the optimizer's contribution.
- Monitor the first 3-5 experiments live to confirm:
  - No OOM crashes on the M4 Max's 128GB unified memory.
  - Per-step timing is within 5% of AdamW-only at the same scale.
  - val_bpb is trending lower than the AdamW-only baseline at target scale.
  - The agent is exploring both Muon hyperparameters AND architecture changes (not just one dimension).
- Key overnight questions to answer:
  - Does the agent keep Muon at DEPTH=6, AR=64? (Expected: yes)
  - If the agent scales down to AR=32, does it also disable Muon? (Expected: yes, per issue #21 pattern)
  - What is the best val_bpb achieved, and does it beat the M5 Max's 1.235?
  - Does the agent discover the gradual LR reduction pattern the M5 Max found (0.04 → 0.005)?
- Let it run overnight. In the morning, review `results.tsv` and the git log.

**5e. Results Writeup and PR (1 hour)**

- Update `results.tsv` with the overnight run data.
- Update `README.md` to document Muon support, replace "AdamW only" with the dual-optimizer description, and add M4 Max benchmark results.
- Open a PR to `trevin-creator/autoresearch-mlx` with the full diff and overnight results.
- Optionally cross-post findings to the `karpathy/autoresearch` discussions, since Karpathy links both Mac forks in the main README.

**Total estimated effort:** 10-17 hours (plus overnight wall-clock for the autonomous run).

---

## 8. Future Work

- **Crossover map:** After the overnight run, plot val_bpb vs model scale (AR) for both Muon and AdamW-only configs. Identify the exact AR where Muon begins to win on M4 Max. Publish this as a reference for other Apple Silicon users.
- **Newton-Muon:** Add right-preconditioning by the input second-moment matrix for ~4% additional wall-clock savings (per arXiv 2604.01472). Requires maintaining a running estimate of `Z·Zᵀ` and periodic Cholesky inverse — feasible in MLX but adds complexity.
- **Quintic NS coefficients:** The `stockeh/mlx-optimizers` library already supports `quintic` and `polar_express` coefficient sets via a backend parameter. Evaluate whether these converge in fewer steps on Metal and port the best-performing variant.
- **`mx.fast` integration:** If Apple adds a fused NS-like op to `mx.fast` (analogous to their RMSNorm and attention kernels), swap in for zero-overhead orthogonalization. Monitor the MLX changelog and `picochat` for signals.
- **Turbo-Muon AOL preconditioning:** Spectral preconditioning to reduce NS steps from 5 to 2-3 while maintaining orthogonality quality.
- **Cross-fork parity testing:** Run identical model configs on the CUDA upstream and this MLX fork (both with Muon) to measure the remaining gap attributable purely to hardware throughput vs optimizer differences.
- **Upstream to `mlx-optimizers`:** If the inline Muon implementation proves stable, extract it as a contribution to `stockeh/mlx-optimizers` with autoresearch-specific defaults and `mx.compile` wrapping as a worked example.
- **Goyal fork sync:** Merge Naman Goyal's improvements (linear weight decay schedule, eval batch size fix, `__main__` guard) alongside Muon. These are orthogonal improvements that compound — his fork reports 2-5% BPB improvement from linear weight decay alone.

---

## 9. References

### 9.1 MLX Muon Implementations (read these first)

- scasella. `nanochat-mlx` — Full MLX port of nanochat with working Muon+AdamW in `optim.py`. [GitHub](https://github.com/scasella/nanochat-mlx). **Primary reference for MLX-specific Muon patterns.**
- stockeh. `mlx-optimizers` — Pip-installable MLX optimizer library with a Muon class, Newton-Schulz backends, and automatic AdamW fallback for non-2D params. [GitHub](https://github.com/stockeh/mlx-optimizers) · [Docs](https://stockeh.github.io/mlx-optimizers/build/html/_autosummary/mlx_optimizers.Muon.html)
- Goekdeniz-Guelmez. "Adding support for the Muon Optimizer" — PR #1914 to MLX core (redirected to mlx-optimizers). Discussion covers tensor reshaping, NS formula variants, and Moonshot scaling. [GitHub PR](https://github.com/ml-explore/mlx/pull/1914)

### 9.2 MLX Training Infrastructure (for idiomatic patterns)

- Hannun, A. `picochat` — Smaller/faster nanochat in MLX, by an MLX core team member. Reference for idiomatic MLX training loops, `mx.compile` usage, and Metal memory management. [GitHub](https://github.com/awni/picochat)
- Apple MLX Team. `mlx` — The framework itself. Key APIs: `mx.compile`, `mx.linalg.norm`, `mx.metal.get_peak_memory()`, lazy evaluation model. [GitHub](https://github.com/ml-explore/mlx) · [Docs](https://ml-explore.github.io/mlx/build/html/index.html)
- trevin-creator. `autoresearch-mlx` — Target repo for this work. [GitHub](https://github.com/trevin-creator/autoresearch-mlx)
- Goyal, N. `autoresearch` (MLX fork) — Independent MLX port with improvements over trevin-creator's fork (eval batch sizing, linear weight decay schedule, `__main__` guard). Notes Muon as unported. [GitHub](https://github.com/thenamangoyal/autoresearch) · [Blog post](https://namangoyal.com/blog/2026/autoresearch-mlx/)

### 9.3 Canonical Muon References (what you're translating from)

- Jordan, K. `Muon` — The original reference implementation (~200 lines). Source of truth for the algorithm, NS coefficients `(3.4445, -4.7750, 2.0315)`, and parameter classification logic. [GitHub](https://github.com/KellerJordan/Muon/blob/master/muon.py) · [Blog post](https://kellerjordan.github.io/posts/muon/)
- PyTorch. `torch.optim._muon` — Muon in PyTorch core (v2.9.0+). Most thoroughly reviewed implementation; clean handling of tall-skinny transpose and quintic computation. [GitHub](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py)
- Keras. `keras.src.optimizers.muon` — Framework-agnostic Muon with auto-exclusion of embeddings and configurable layer exclusion list. Useful for cross-checking parameter classification logic. [GitHub](https://github.com/keras-team/keras/blob/v3.12.0/keras/src/optimizers/muon.py)
- Karpathy, A. `autoresearch` — Upstream repo. Shows how Muon+AdamW are wired together in the training loop the agent modifies. [GitHub](https://github.com/karpathy/autoresearch)

### 9.4 Muon Theory & Optimization Research

- Bernstein, J. "Deriving Muon." [Blog post](https://jeremybernste.in/writing/deriving-muon)
- MoonshotAI. "Muon is Scalable for LLM Training." [arXiv:2502.16982](https://arxiv.org/abs/2502.16982). LR adjustment formula for matching AdamW RMS, referenced in PyTorch core and mlx-optimizers implementations.
- Chen et al. "Newton-Muon." [arXiv:2604.01472](https://arxiv.org/html/2604.01472v1), April 2026. Right-preconditioning extension for ~4% wall-clock savings (future work).
- NVIDIA. "Emerging Optimizers: Muon." [Docs](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html)

### 9.5 Community Results & Empirical Findings

- hirokimsd. "M5 Max 128GB results: val_bpb 1.574 → 1.235 (25 experiments)." Issue #21 on autoresearch-mlx. **Critical finding: Muon lost to AdamW on M5 Max with small model; Muon won on Mac Mini.** Hardware-specific crossover is real. [GitHub Issue](https://github.com/trevin-creator/autoresearch-mlx/issues/21)
- SkyPilot. "Scaling Karpathy's Autoresearch." — Parallel autoresearch on 16 GPUs. Key finding: Muon hyperparams (`muon_beta2=0.98`, NS steps, matrix LR) were among the late-stage improvements the agent found after exhausting architecture search. AR=96 was the sweet spot on H100. [Blog](https://blog.skypilot.co/scaling-autoresearch/)

### 9.6 Performance Optimization References

- nil0x9. `flash-muon` — Custom CUDA kernel for symmetric `X @ X.T` matmul that dominates NS cost. Useful for understanding the computational bottleneck even though the kernel won't port to Metal. [GitHub](https://github.com/nil0x9/flash-muon)
- miolini. `autoresearch-macos` — PyTorch+MPS Mac fork. Shows what MPS-specific workarounds look like (disabled `torch.compile`, lowered batch sizes, cast optimizer states). Contrast with native MLX approach. [GitHub](https://github.com/miolini/autoresearch-macos)
