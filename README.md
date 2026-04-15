# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled through `program.md`. This port keeps the same basic rules: one mutable `train.py`, one metric (`val_bpb`), a fixed 5-minute training budget, and keep-or-revert via git. It runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx), so there is no PyTorch or CUDA dependency.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data + tokenizer prep
uv run prepare.py

# run one 5-minute training experiment
uv run train.py
```

Then point Claude Code or another coding agent at `program.md` and let it run the loop.

## What matters

- `prepare.py` - data prep, tokenizer, dataloader, and evaluation. Treat as fixed.
- `train.py` - model, optimizer, and training loop. This is the file the agent edits.
- `program.md` - the autonomous experiment protocol.
- `results.tsv` - logged experiment history.

The loop is the same as upstream: edit `train.py`, run a fixed-budget experiment, read `val_bpb`, keep the change if it wins, revert if it loses, and repeat.

## Public baseline results

The public `results.tsv` captures the initial hardware-local walk from the default baseline down to `1.807902`:

| Commit | val_bpb | Status | Description |
|---|---:|---|---|
| `383abb4` | 2.667000 | keep | baseline (AdamW, default config) |
| `909dd59` | 2.588904 | keep | halve total batch size to `2^16` |
| `4161af3` | 2.533728 | keep | increase matrix LR to `0.04` |
| `5efc7aa` | 1.807902 | keep | reduce depth from `8` to `4` |

That result already shows the core Apple Silicon pattern: with a fixed 5-minute wall clock, smaller faster-training models can beat larger ones simply by fitting more optimizer steps into the budget.

## Longer Apple Silicon runs

Longer overnight runs on the working MLX port pushed much further. The long Mac Mini test is included here because it found a meaningfully different winner stack from the Max-class machines.

| Machine | Current best | Starting point | Repeated wins |
|---|---:|---:|---|
| M4 Max #1 | 1.294526 | 1.596971 | AdamW-only, low matrix LR, 3x MLP, no logit cap, moderate weight decay |
| M4 Max #2 | 1.330509 | 1.807902 | leaner batch, long anneal, SiLU, lower regularization, no logit cap |
| Mac Mini (long run) | 1.353329 | 1.922472 | Muon, sharper attention, smaller MLP, lower scalar LR |

The Mac Mini result matters because it did not just rediscover the same exact recipe. On smaller Apple Silicon hardware, the strongest changes leaned toward more aggressive step-efficiency wins. Later transfer tests showed some of those Mac Mini findings did not carry cleanly onto the Max baseline, which is exactly the kind of hardware-specific behavior this loop is useful for uncovering.

## Muon Newton-Schulz micro-benchmarks (M4 Max Mac Studio, 128GB)

Hardware: **Mac Studio M4 Max, 16-core CPU / 40-core GPU / 128GB unified memory**. Numbers are wall-clock ms per 5 Newton-Schulz iterations at various matrix sizes, averaged over 10 runs.

| Shape | bfloat16 | float32 |
|---|---:|---:|
| 256 × 256 | 0.41 ms | 0.39 ms |
| 384 × 384 | 0.58 ms | 0.57 ms |
| 512 × 512 | 0.81 ms | 0.88 ms |
| 768 × 768 | 1.66 ms | 1.79 ms |

**Per-step Muon overhead at target config** (12 weight matrices × 512², 5 NS iterations): **~1.0 ms per training step**. Over a 5-minute budget at 400–700 steps/run, that's **0.6–0.7s of total NS overhead** — well under the PRD's 5% budget.

**Two M4 Max findings worth recording:**
- **Do not wrap `newton_schulz` in `mx.compile`.** On this hardware the wrapped version is ~**0.62× speed** (38% *slower*) — likely JIT warmup cost not amortizing over the small call count. The code as shipped on `muon-mlx` deliberately does not use `mx.compile` for NS.
- **`bfloat16` ties or beats `float32`** at every matrix size tested. On M1-class silicon the picture was mixed (float32 sometimes won). Stick with the default `MUON_NS_DTYPE = "bfloat16"` here.

End-to-end training val_bpb comparisons (AdamW-only vs Muon+AdamW at `DEPTH=6, ASPECT_RATIO=64`) pending — see `docs/superpowers/plans/2026-04-14-muon-mlx.md`.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **Dual-path optimizer.** Uses a dual-path optimizer: **Muon** (Newton-Schulz orthogonalized momentum) for 2D block weight matrices, **AdamW** for embeddings, the output head, layer norms, and scalar mixers. See `prd_muon_mlx.md` for the port rationale and Apple Silicon model-size crossover analysis.
- **Smaller eval token budget.** Reduced for faster iteration on Apple Silicon while keeping the same `evaluate_bpb` interface in `prepare.py`.
- **Roughly 6-7 minutes per experiment.** Expect 5 minutes of training plus compile and eval overhead.
- **MFU reporting is placeholder.** There is no Apple Silicon equivalent to the H100 FLOPs reference used upstream.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) - autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) - original MLX port of autoresearch (this fork extends it with Muon)
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) - MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) - MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. See [LICENSE](LICENSE).
