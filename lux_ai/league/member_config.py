"""
Loading and validation of pool-member model configs.

Frozen agent configs predate some model kwargs that were added to the trainer
later, so absent optional keys are filled with the value the trainer used at
the time that config was written (verified against the checkpoints in
internal_testing/). Members whose weights cannot actually be loaded are
rejected at admission - an actor process dies silently on an unhandled
exception, so a bad member must never reach one.
"""
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import torch
from omegaconf import OmegaConf

from ..utils import flags_to_namespace

# NB: sum_player_embeddings was added after the 2021-10 agents were trained;
# those checkpoints have the summed (352-channel) embedding merger, so True is
# the correct historical default for configs missing the key.
OPTIONAL_MODEL_KEY_DEFAULTS = {
    "n_merge_layers": 1,
    "sum_player_embeddings": True,
    "use_index_select": False,
    "normalize": False,
    "rescale_value_input": True,
    "rescale_se_input": True,
    # Ported with the team-g adaptive-embedding change: configs written before it
    # (every agent in internal_testing/) must keep building a uniform embedding_dim.
    "adaptive_embedding_dims": False,
    "adaptive_embedding_max_values": None,
    "obs_space_kwargs": {},
    "reward_space_kwargs": {},
}


def load_member_flags(config_path: Union[str, Path]) -> SimpleNamespace:
    """Load a pool member's config.yaml as a namespace usable by _create_model."""
    member_conf = OmegaConf.to_container(OmegaConf.load(str(config_path)))
    for key, default in OPTIONAL_MODEL_KEY_DEFAULTS.items():
        member_conf.setdefault(key, default)
    return flags_to_namespace(member_conf)


def validate_member(config_path: Union[str, Path], checkpoint_path: Union[str, Path]) -> Optional[str]:
    """
    Build the member's model on CPU and load its weights.
    Returns None if the member is usable, or a human-readable reason if not.

    Channel counts come from the member's own obs space, which is what the
    prefixed input layer reads out of the env dict, so validating against it is
    faithful to what the actor will build.
    """
    from ..nns import create_opponent_model

    try:
        member_flags = load_member_flags(config_path)
    except Exception as e:
        return f"config is not loadable as model flags ({e})"
    try:
        model = create_opponent_model(
            member_flags,
            torch.device("cpu"),
            obs_space=member_flags.obs_space(**member_flags.obs_space_kwargs),
            obs_space_prefix="",
        )
    except Exception as e:
        return f"model could not be constructed ({e})"
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    except Exception as e:
        first_error = str(e).splitlines()[1].strip() if len(str(e).splitlines()) > 1 else str(e)
        return f"weights do not match the model built from its config ({first_error})"
    finally:
        del model
    logging.debug(f"League: validated member config {config_path}")
    return None
