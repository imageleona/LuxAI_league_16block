from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments

import hydra
import logging
import os
import sys
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
from lux_ai.torchbeast.monobeast import train


os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})
    # None (the default) leaves the engine picking from {12, 16, 24, 32} per seed,
    # exactly as before. An int pins every episode to that square map size.
    flags.setdefault("map_size", None)
    # Compact embeddings for low-cardinality features (team-g). False reproduces
    # the original uniform-width behaviour exactly.
    flags.setdefault("adaptive_embedding_dims", False)
    flags.setdefault("adaptive_embedding_max_values", None)

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 10.)
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)

    # Model params
    flags.setdefault("use_index_select", True)
    if flags.get("use_index_select"):
        logging.info("index_select disables padding_index and is equivalent to using a learnable pad embedding.")

    # Reloading previous run params
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_value_warmup_batches", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="resume_config")
def main(flags: DictConfig):
    # NB: not OmegaConf.from_cli() - that reads raw sys.argv, so hydra's own
    # flags (--config-name, --multirun, ...) become config keys and the merges
    # below fail with "Key '--config-name' is not in struct". Only reachable when
    # resuming, since those merges sit behind the load_dir branch. Keep just the
    # key=value overrides, dropping hydra's '+' prefix for appended keys.
    cli_conf = OmegaConf.from_dotlist([
        arg.lstrip("+") for arg in sys.argv[1:]
        if "=" in arg and not arg.startswith("--")
    ])
    if Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    logging.info(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            # NB: not vars(flags) - flags is a DictConfig, whose __dict__ holds
            # OmegaConf's internals rather than the settings. Current wandb
            # tries to expand one of those (a dataclass holding a defaultdict)
            # and dies with "first argument must be callable or None".
            config=OmegaConf.to_container(flags, resolve=True),
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags, resolve=True))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
