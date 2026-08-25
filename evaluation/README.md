# Phase 1 evaluation harness

The harness runs the official Node engine once per concurrent game while sharing one
PyTorch model per unique agent hash. At each turn it batches every active state for
that model into one forward pass. Engine steps run concurrently.

The fixed full evaluation uses seeds `0..49`, all four map sizes, and both player
seats: 50 x 4 x 2 = 400 games.

```bash
python -m evaluation.phase1 self-test \
  --agent internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research \
  --output evaluation_runs/baseline_selftest \
  --parallel-games 8
```

Compare a packaged candidate with the baseline:

```bash
python -m evaluation.phase1 compare \
  --candidate path/to/candidate_package \
  --output evaluation_runs/candidate_vs_baseline \
  --parallel-games 8
```

`summary.json` and `summary.md` contain aggregate, seat, map-size, confidence-
interval, city-margin, and timing results. Per-game diagnostics are always retained.
Lux replay files are retained only when the logical candidate loses.

The primary win rate scores ties as one half. Its 95% interval is the Wilson score
interval; this convention is recorded in every summary.
