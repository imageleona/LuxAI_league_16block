"""
Actor-process side of the league.

Runs inside `act()` workers: keeps the published league state fresh, assigns
an opponent + seat per episode, runs frozen opponent models for their seat and
splices their actions into the executed action tensors, and extracts episode
outcomes for reporting.

No imports from lux_ai.torchbeast - only model construction (lux_ai.nns) and
env obs-space helpers (lux_ai.lux_gym).
"""
import logging
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .flags import LeagueFlags
from .member_config import load_member_flags
from .pool import PoolMember
from .state_io import read_state_if_newer
from ..lux_gym import create_flexible_obs_space
from ..nns import create_opponent_model

MIRROR = "mirror"
LEAGUE = "league"


@dataclass
class EnvAssignment:
    mode: str = MIRROR
    member: Optional[PoolMember] = None
    learner_seat: int = 0

    @property
    def opponent_seat(self) -> int:
        return 1 - self.learner_seat


def _nested_index(x: Union[Dict, torch.Tensor], idx: torch.Tensor) -> Union[Dict, torch.Tensor]:
    if isinstance(x, dict):
        return {key: _nested_index(val, idx) for key, val in x.items()}
    return x[idx]


class OpponentModelCache:
    """LRU cache of frozen opponent models resident on the actor device."""

    def __init__(self, flags: SimpleNamespace, teacher_flags: Optional[SimpleNamespace], max_size: int):
        self.device = flags.actor_device
        self.max_size = max(1, max_size)
        # Opponent models consume the training env's observations, so they are
        # built against the env obs space (+ the member's obs prefix), exactly
        # like the student/teacher models are.
        self.env_obs_space = create_flexible_obs_space(flags, teacher_flags)
        self._models: "OrderedDict[str, torch.nn.Module]" = OrderedDict()

    def get(self, member: PoolMember) -> torch.nn.Module:
        model = self._models.get(member.member_id)
        if model is not None:
            self._models.move_to_end(member.member_id)
            return model
        model = self._load(member)
        self._models[member.member_id] = model
        while len(self._models) > self.max_size:
            evicted_id, evicted_model = self._models.popitem(last=False)
            del evicted_model
            logging.info(f"League actor: unloaded opponent model {evicted_id}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return model

    def _load(self, member: PoolMember) -> torch.nn.Module:
        try:
            member_flags = load_member_flags(member.config_path)
            model = create_opponent_model(
                member_flags,
                self.device,
                obs_space=self.env_obs_space,
                obs_space_prefix=member.obs_space_prefix,
            )
            checkpoint = torch.load(member.checkpoint_path, map_location=torch.device("cpu"))
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:
            # Actor threads/processes die silently on unhandled exceptions, so
            # always leave a trace of which member broke.
            logging.exception(f"League actor: FAILED to load opponent model {member.member_id} "
                              f"from {member.checkpoint_path}")
            raise
        logging.info(f"League actor: loaded opponent model {member.member_id}")
        return model


class ActorLeagueClient:
    def __init__(self, flags: SimpleNamespace, teacher_flags: Optional[SimpleNamespace], actor_index: int):
        self.league_flags = LeagueFlags.from_dict(flags.league)
        if not self.league_flags.state_path:
            raise ValueError("flags.league['state_path'] must be set before actors start")
        self.n_envs = flags.n_actor_envs
        base_seed = 0 if flags.seed is None else int(flags.seed)
        self.rng = np.random.RandomState((base_seed + 7919 * (actor_index + 1)) % (2 ** 31))
        self.cache = OpponentModelCache(flags, teacher_flags, self.league_flags.max_loaded_opponents)

        self._version = -1
        self._members: List[PoolMember] = []
        self._probs: np.ndarray = np.zeros(0)
        self.assignments: List[EnvAssignment] = [EnvAssignment() for _ in range(self.n_envs)]
        # (n_envs, 2) float mask: 1 for seats controlled by the learning agent.
        self.mask = torch.ones((self.n_envs, 2), dtype=torch.float32)
        self.refresh_state()

    # ------------------------------------------------------------ state sync

    def refresh_state(self) -> None:
        state = read_state_if_newer(self.league_flags.state_path, self._version)
        if state is None:
            return
        self._version = state["version"]
        self._members = [PoolMember.from_dict(m) for m in state["members"]]
        probs = np.asarray(state["probs"], dtype=np.float64)
        if len(probs) > 0 and probs.sum() > 0:
            probs = probs / probs.sum()
        self._probs = probs

    # ------------------------------------------------------------ assignment

    def assign(self, env_idx: int) -> EnvAssignment:
        if len(self._members) == 0 or self.rng.rand() < self.league_flags.mirror_frac:
            assignment = EnvAssignment(mode=MIRROR)
            self.mask[env_idx] = 1.
        else:
            member_idx = int(self.rng.choice(len(self._members), p=self._probs))
            learner_seat = int(self.rng.randint(2))
            assignment = EnvAssignment(mode=LEAGUE, member=self._members[member_idx],
                                       learner_seat=learner_seat)
            self.mask[env_idx] = 0.
            self.mask[env_idx, learner_seat] = 1.
        self.assignments[env_idx] = assignment
        return assignment

    def assign_all(self) -> None:
        self.refresh_state()
        for env_idx in range(self.n_envs):
            self.assign(env_idx)

    def mask_clone(self) -> torch.Tensor:
        return self.mask.clone()

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def apply_opponent_actions(
            self,
            env_output: Dict[str, Union[Dict, torch.Tensor]],
            agent_output: Dict[str, Union[Dict, torch.Tensor]],
    ) -> None:
        """
        For every league-controlled env, overwrite the opponent seat's actions
        in agent_output["actions"] (in place) with the frozen opponent's
        actions, so the env executes - and the buffers record - those actions.
        """
        env_idxs_by_member: "OrderedDict[str, List[int]]" = OrderedDict()
        for env_idx, assignment in enumerate(self.assignments):
            if assignment.mode == LEAGUE:
                env_idxs_by_member.setdefault(assignment.member.member_id, []).append(env_idx)
        if not env_idxs_by_member:
            return

        # Mirrors what DictInputLayer reads, so we slice only what is needed.
        info = env_output["info"]
        model_info = {
            "input_mask": info["input_mask"],
            "available_actions_mask": info["available_actions_mask"],
        }
        if "subtask_embeddings" in info:
            model_info["subtask_embeddings"] = info["subtask_embeddings"]
        model_input_full = {"obs": env_output["obs"], "info": model_info}
        for member_id, env_idxs in env_idxs_by_member.items():
            member = self.assignments[env_idxs[0]].member
            model = self.cache.get(member)
            idx = torch.tensor(env_idxs, dtype=torch.long,
                               device=env_output["done"].device)
            model_input = _nested_index(model_input_full, idx)
            opponent_output = model(model_input, sample=self.league_flags.opponent_sample)
            for slice_idx, env_idx in enumerate(env_idxs):
                seat = self.assignments[env_idx].opponent_seat
                for act_key, action_tensor in agent_output["actions"].items():
                    action_tensor[env_idx, :, seat] = \
                        opponent_output["actions"][act_key][slice_idx, :, seat]

    # --------------------------------------------------------------- outcome

    def handle_dones(self, done: torch.Tensor, cached_reward: torch.Tensor, outcome_queue) -> None:
        """
        Called right after the terminal step (before buffers are filled with a
        new mask): report outcomes for finished league games and re-assign the
        finished envs for their next episode.
        """
        done_idxs = torch.nonzero(done, as_tuple=False).flatten().tolist()
        if not done_idxs:
            return
        for env_idx in done_idxs:
            assignment = self.assignments[env_idx]
            if assignment.mode == LEAGUE:
                # GameResultReward is zero-sum: +1 win / -1 loss / 0 draw.
                reward = float(cached_reward[env_idx, assignment.learner_seat].item())
                outcome = min(max((reward + 1.) / 2., 0.), 1.)
                outcome_queue.put((assignment.member.member_id, outcome))
        self.refresh_state()
        for env_idx in done_idxs:
            self.assign(env_idx)
