# Lux AI Season 1: PFSP league for the 16-block agent

This repository fine-tunes the 16-block, 96-channel Lux AI Season 1 agent with a
Prioritized Fictitious Self-Play (PFSP) league. Half of the games are self-play;
the other half use frozen opponents selected from a checkpoint pool, with more
sampling weight given to opponents that the learner does not yet beat reliably.

Run all commands below from the repository root. The project was verified on
Windows 11 with one NVIDIA GPU (16 GB VRAM), Python 3.7.16, CUDA 11.7, and
Node.js 24.18.0.

## 1. Setup

Install these system tools first:

- [Miniconda or Anaconda](https://docs.conda.io/projects/miniconda/en/latest/)
- [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/)
- [Node.js](https://nodejs.org/) (the Lux game engine is a Node process)
- An NVIDIA driver compatible with CUDA 11.7

Create and activate the Python environment:

```bash
conda create --name lux-league python=3.7.16 pip -y
conda activate lux-league
```

On Windows, always activate the environment instead of calling its
`python.exe` by absolute path. Activation adds Conda's DLL directories to
`PATH` and avoids misleading import failures from packages such as `wandb`.

After cloning the repository in section 2, install PyTorch and the remaining
Python dependencies:

```bash
python -m pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

`requirements.txt` is the short, maintained dependency list. The complete
environment used for the original runs is preserved in
[`league/requirements-freeze.txt`](league/requirements-freeze.txt).

Check the installation:

```bash
python --version
node --version
git lfs version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest lux_ai/league/tests evaluation/tests -q
```

`torch.cuda.is_available()` must print `True` for the supplied GPU config.

## 2. Clone the repository

Replace `<REPOSITORY_URL>` with this repository's Git URL:

```bash
git lfs install
git clone <REPOSITORY_URL> LuxAI_league_16block
cd LuxAI_league_16block
git lfs pull
```

The last command downloads the model checkpoints. Confirm that the starting
16-block weights exist:

```bash
git lfs ls-files
```

The list should include
`league_agents/haruto_16block/lux_ai/rl_agent/40000_weights.pt` and the four
opponents under `internal_testing/` referenced by
`conf/league_haruto_16block.yaml`.

ADD THESE FOLDERS FROM GOOGLE DRIVE IN THE MAIN FOLDER:
https://drive.google.com/drive/folders/14anheutHIDDFXqNTtgl8poq0MC3C_AKA?usp=sharing


## 3. Code guide and running the league

### What each part does

| Path | Purpose |
| --- | --- |
| `run_monobeast.py` | Main training command. Loads a Hydra config, initializes logging, and starts the learner. |
| `conf/league_haruto_16block.yaml` | Default 16-block experiment: model shape, optimizer, devices, PFSP settings, starting checkpoint, and evaluation schedule. |
| `lux_ai/torchbeast/monobeast.py` | Actor/learner training loop, loss calculation, checkpointing, and league integration. |
| `lux_ai/league/` | Opponent pool, PFSP sampling, win-rate tracking, snapshots, and resumable league state. |
| `lux_ai/lux_gym/` | Lux environment plus observation, action, and reward spaces. |
| `lux_ai/nns/` | Neural-network blocks and model construction. |
| `lux_ai/rl_agent/` | Agent inference and checkpoint-loading code. |
| `league_agents/haruto_16block/` | The 16-block starting policy and its model configuration. |
| `internal_testing/` | Frozen opponent checkpoints used as permanent league anchors. |
| `evaluation/` | Fixed-seed evaluation and league-run comparison tools. |
| `outputs/` | Hydra run directories containing the resolved config, checkpoints, league state, snapshots, and evaluation logs. |

More detail about the league internals is in
[`lux_ai/league/README.md`](lux_ai/league/README.md).

### Start the 16-block league

Activate the environment, enter the repository root, and run:

```bash
conda activate lux-league
python run_monobeast.py --config-name league_haruto_16block
```

The default preset trains for 500,000 environment steps on `cuda:0`, samples
random map sizes from 12, 16, 24, and 32, and writes results to
`outputs/<MM-DD>/<HH-MM-SS>/`. Weights & Biases logging is enabled; run
`wandb login` first, or disable it from the command line as shown below.

For a short smoke test:

```bash
python run_monobeast.py --config-name league_haruto_16block \
  total_steps=640 league.anchor_eval_enabled=false disable_wandb=true
```

PowerShell accepts the command on one line. For a multiline PowerShell command,
replace `\` with the PowerShell continuation character (a backtick).

### CLI options

The command uses [Hydra](https://hydra.cc/), so there are two kinds of CLI
arguments:

- `--config-name league_haruto_16block` selects a YAML file from `conf/`
  without the `.yaml` suffix.
- `key=value` overrides a top-level config value.
- `league.key=value` overrides a value inside the `league` section.
- `--cfg job --resolve` prints the resolved configuration and exits without
  training.
- `--multirun` launches a Hydra parameter sweep; avoid it until one normal run
  has completed successfully.

Common overrides:

| Override | Meaning |
| --- | --- |
| `total_steps=1000000` | Change the training budget. |
| `actor_device=cuda:1 learner_device=cuda:1` | Use a different GPU. Keep both devices together for the standard single-GPU run. |
| `disable_wandb=true` | Run without Weights & Biases. Files are still written locally. |
| `league.anchor_eval_enabled=false` | Disable periodic fixed-seed evaluation to shorten a test run. |
| `league.anchor_eval_every_n_games=1000` | Evaluate less often. |
| `league.mirror_frac=0.25` | Use self-play for 25% of games and pool opponents for 75%. |
| `league.priority=uniform` | Replace PFSP weighting with uniform opponent sampling. |
| `checkpoint_freq=30` | Save a regular checkpoint every 30 minutes. |

Preview an override before committing GPU time:

```bash
python run_monobeast.py --config-name league_haruto_16block \
  total_steps=1000000 disable_wandb=true --cfg job --resolve
```

### Resume an interrupted run

Use a full checkpoint such as `<step>.pt`, not a weights-only
`<step>_weights.pt` file:

```bash
python run_monobeast.py --config-name league_haruto_16block \
  load_dir=outputs/<MM-DD>/<HH-MM-SS> \
  checkpoint_file=<step>.pt weights_only=false total_steps=1000000
```

This restores the model, optimizer, learning-rate scheduler, step counter, and
`league/state.json`. When `weights_only=true`, training starts a new run from
the selected policy weights instead.

The original Toad Brigade code is MIT licensed; see [`LICENCE.txt`](LICENCE.txt).
