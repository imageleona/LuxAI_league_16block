# Model weights

None of the `.pt` checkpoints are distributed through this git repository. There are 13
unique checkpoints at roughly 80 MB each — about **1.04 GB**, which exceeds GitHub's
free Git-LFS allowance (1 GB storage, 1 GB/month bandwidth).

Every hash below is a **SHA-256 of the file content**, so you can verify a download
regardless of where it came from.

```bash
sha256sum <file>                      # Linux / Git Bash
certutil -hashfile <file> SHA256      # Windows
```

---

## Where to get them

### Upstream agents — `internal_testing/`

These are the Toad Brigade team's released checkpoints, published in the upstream
repository via Git-LFS:

```bash
git clone https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 upstream
# then copy upstream/internal_testing/ over this repo's internal_testing/
```

### League-run agents — `league_agents/`

Produced by the experiments in [RESULTS.md](RESULTS.md). Hosted separately.

> **TODO — fill this in after uploading.** These files are not yet published anywhere.
> Once they are (Google Drive folder, Kaggle dataset, or a GitHub Release on your fork),
> replace this note with the link. A GitHub Release is usually the least friction: the
> per-file limit is 2 GB and release assets do not consume LFS quota.

---

## Manifest

Sizes are bytes. Duplicate hashes are marked — they are byte-identical files, so you
only need to download the content once.

### `internal_testing/` — upstream

| Agent | Weights file | SHA-256 | Size |
|---|---|---|---|
| `hall_of_fame/11-24_12-56-23_062179520_must_research` | `062179520_weights.pt` | `40248f0f…7cdffd22` | 79,552,808 |
| `hall_of_fame/11-09_21-32-04_59822400` | `59822400_weights.pt` | `4215bc23…f22c0dc1` | 79,552,808 |
| `hall_of_fame/10-10_11-18-12_28576448` | `28576448_weights.pt` | `4e153c77…eddd8580` | 79,438,120 |
| `hall_of_fame/10-10_11-18-12_28576448_must_research` | `28576448_weights.pt` | `4e153c77…eddd8580` ⟵ dup | 79,438,120 |
| `hall_of_fame/09-17_22-05-30_20000128` | `20000128_weights.pt` | `62169658…b2a8e37f` | 79,853,014 |
| `hall_of_fame/09-07_01-44-10_10088000` | `10088000_weights.pt` | `57aa4b8a…ee796c6d` | 79,854,550 |
| `internal_agents/10-02_11-29-02_20000192` | `20000192_weights.pt` | `3633cf36…9ac88e20` | 79,438,120 |
| `internal_agents/10-08_17-35-45_20000192` | `20000192_weights.pt` | `57b66dd7…e0933a4d` | 79,438,120 |

### `league_agents/` — produced by these experiments

| Agent | Weights file | SHA-256 | Size |
|---|---|---|---|
| `pfsp_final` | `1000192_weights.pt` | `db06b524…47cf213b` | 79,550,071 |
| `sp_final` | `1000192_weights.pt` | `17ff18fd…ea44c520` | 79,550,071 |
| `pfsp_5m_seed7` | `5000192_weights.pt` | `f460d4e5…236d37db` | 79,550,071 |
| `pfsp_5m_seed7_peak961k` | `000000961536_weights.pt` | `919cb0c8…b89cf18d` | 79,529,590 |
| `diag_lr5e5_200k` | `200192_weights.pt` | `6a788554…0e76ace8` | 79,549,892 |
| `0818_teamG` | `1000192_weights.pt` | `17ff18fd…ea44c520` ⟵ dup of `sp_final` | 79,550,071 |
| `0818_teamG_5mfinal` | `5000192_weights.pt` | `f460d4e5…236d37db` ⟵ dup of `pfsp_5m_seed7` | 79,550,071 |
| `0818_teamG_peak961k` | `000000961536_weights.pt` | `919cb0c8…b89cf18d` ⟵ dup of `pfsp_5m_seed7_peak961k` | 79,529,590 |

Full hashes are in [`weights.sha256`](weights.sha256), in `sha256sum -c` format.

**5 unique league agents, not 8.** The `0818_teamG*` directories are full Kaggle
submission packages — each vendors its own copy of `lux_ai/` alongside the weights —
while the others hold only the minimal frozen-agent layout the evaluation harness loads.
The weights themselves are identical.

---

## Required layout

Each agent directory must look like this, with **exactly one** `.pt` file:

```
<agent_dir>/lux_ai/rl_agent/
    config.yaml               <- the model config it was trained with
    rl_agent_config.yaml
    <step>_weights.pt
```

`evaluation.harness.AgentSpec.from_directory` and `lux_ai.league.member_config` both
depend on this. A second `.pt` in the directory will fail the pool-preset test.

## Which weights do you actually need?

| Goal | Required |
|---|---|
| Run the tests | `internal_testing/hall_of_fame/{11-24…, 10-10…}`, `internal_testing/internal_agents/{10-02…, 10-08…}` |
| Train any `league_*` arm | The four above (teacher + 2 anchors + 2 initial members) |
| Reproduce the held-out evaluation | The above, plus `league_agents/{pfsp_final, sp_final}` and `internal_testing/hall_of_fame/{11-09…, 09-17…}` |

`hall_of_fame/09-07_01-44-10_10088000` is listed for completeness but **cannot be
rebuilt** by the current code — its config sets an `obs_space_kwarg` the obs space no
longer accepts. It is unusable as a pool member or an eval opponent.
