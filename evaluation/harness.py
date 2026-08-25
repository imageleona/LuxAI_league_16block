from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from subprocess import PIPE, Popen
from threading import Thread
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lux_ai.lux.game_constants import GAME_CONSTANTS
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS
from lux_ai.lux_gym.lux_env import LuxEnv, _enqueue_output
from lux_ai.nns import create_model
from lux_ai.league.member_config import OPTIONAL_MODEL_KEY_DEFAULTS
from lux_ai.nns.models import DictActor
from lux_ai.rl_agent import data_augmentation
from lux_ai.utility_constants import DN_CYCLE_LEN, MAX_BOARD_SIZE, MAX_RESEARCH
from lux_ai.utils import flags_to_namespace


HARNESS_VERSION = "phase1-v1"
DEFAULT_MAP_SIZES = (12, 16, 24, 32)
DEFAULT_SEEDS = tuple(range(50))
DEFAULT_BASELINE_DIR = (
    Path(__file__).resolve().parents[1]
    / "internal_testing"
    / "hall_of_fame"
    / "11-24_12-56-23_062179520_must_research"
)
_HASHED_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".pt"}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    temporary.replace(path)


def _content_hash(directory: Path, extra: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _HASHED_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AgentSpec:
    """Files and inference rules that define one evaluated agent."""

    name: str
    directory: Path
    model_config: Path
    agent_config: Path
    checkpoint: Path
    device: str = "cuda:0"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        name: Optional[str] = None,
        device: str = "cuda:0",
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "AgentSpec":
        directory = Path(directory).expanduser().resolve()
        rl_dir = directory / "lux_ai" / "rl_agent"
        model_config = rl_dir / "config.yaml"
        agent_config = rl_dir / "rl_agent_config.yaml"
        checkpoints = sorted(rl_dir.glob("*.pt"))
        missing = [path for path in (model_config, agent_config) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing agent files: {}".format(", ".join(map(str, missing))))
        if len(checkpoints) != 1:
            raise ValueError("Expected exactly one checkpoint in {}, found {}".format(rl_dir, len(checkpoints)))
        overrides_dict = dict(overrides or {})
        agent_hash = _content_hash(directory, {"overrides": overrides_dict, "harness": HARNESS_VERSION})
        return cls(
            name=name or directory.name,
            directory=directory,
            model_config=model_config,
            agent_config=agent_config,
            checkpoint=checkpoints[0],
            device=device,
            overrides=overrides_dict,
            content_hash=agent_hash,
        )


@dataclass(frozen=True)
class MatchRequest:
    agent_0: AgentSpec
    agent_1: AgentSpec
    seed: int
    map_size: int
    candidate_seat: int = 0
    match_id: str = ""

    def __post_init__(self) -> None:
        if self.candidate_seat not in (0, 1):
            raise ValueError("candidate_seat must be 0 or 1")
        if self.map_size not in DEFAULT_MAP_SIZES:
            raise ValueError("map_size must be one of {}".format(DEFAULT_MAP_SIZES))


@dataclass
class MatchResult:
    match_id: str
    seed: int
    map_size: int
    agent_0: str
    agent_1: str
    agent_0_hash: str
    agent_1_hash: str
    candidate_seat: int
    winner: Optional[int]
    final_city_tiles: Tuple[int, int]
    final_units: Tuple[int, int]
    ending_turn: int
    wall_seconds: float
    replay_path: Optional[str]
    diagnostics_path: str
    cache_key: str
    error: str = ""

    @property
    def candidate_score(self) -> float:
        if self.winner is None:
            return 0.5
        return float(self.winner == self.candidate_seat)

    @property
    def candidate_city_margin(self) -> int:
        opponent = 1 - self.candidate_seat
        return self.final_city_tiles[self.candidate_seat] - self.final_city_tiles[opponent]

    @property
    def seconds_per_step(self) -> float:
        return self.wall_seconds / max(self.ending_turn, 1)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["candidate_score"] = self.candidate_score
        result["candidate_city_margin"] = self.candidate_city_margin
        result["seconds_per_step"] = self.seconds_per_step
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchResult":
        fields = {
            key: value[key]
            for key in cls.__dataclass_fields__
            if key in value
        }
        fields["final_city_tiles"] = tuple(fields["final_city_tiles"])
        fields["final_units"] = tuple(fields["final_units"])
        return cls(**fields)


@dataclass
class RunnerStats:
    wall_seconds: float = 0.0
    model_seconds: float = 0.0
    engine_seconds: float = 0.0
    inference_calls: int = 0
    inferred_states: int = 0
    total_turns: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["throughput_seconds_per_step"] = self.wall_seconds / max(self.total_turns, 1)
        result["steps_per_wall_second"] = self.total_turns / max(self.wall_seconds, 1e-12)
        result["mean_inference_batch"] = self.inferred_states / max(self.inference_calls, 1)
        return result


def _prepare_diagnostic_engine(output_dir: Path) -> Path:
    """Copy and instrument the bundled engine to emit exact night-loss fuel data."""
    try:
        from kaggle_environments.envs.lux_ai_2021.lux_ai_2021 import dir_path
    except Exception as exc:
        raise RuntimeError("The Kaggle Lux AI 2021 environment is not installed") from exc

    source_dir = Path(dir_path) / "dimensions"
    source_main = source_dir / "main.js"
    source_bytes = source_main.read_bytes()
    engine_hash = hashlib.sha256(source_bytes).hexdigest()[:16]
    target_dir = output_dir / ".engine" / engine_hash
    target_main = target_dir / "main.js"
    if target_main.is_file():
        return target_main

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_dir.with_name(target_dir.name + ".tmp-{}".format(os.getpid()))
    if temporary.exists():
        shutil.rmtree(str(temporary))
    shutil.copytree(str(source_dir), str(temporary))

    text = (temporary / "main.js").read_text(encoding="utf-8")
    old_night = (
        "handleNight(t){const e=t.game;e.cities.forEach((t=>{"
        "t.fuel<t.getLightUpkeep()?e.destroyCity(t.id):t.fuel-=t.getLightUpkeep()}))"
    )
    new_night = (
        "handleNight(t){const e=t.game;e.__phase1NightLosses=[],e.cities.forEach((t=>{"
        "t.fuel<t.getLightUpkeep()?(e.__phase1NightLosses.push({turn:e.state.turn,team:t.team,"
        "cityid:t.id,fuel:t.fuel,upkeep:t.getLightUpkeep(),shortfall:t.getLightUpkeep()-t.fuel,"
        "tiles:t.citycells.map((t=>[t.pos.x,t.pos.y]))}),e.destroyCity(t.id)):"
        "t.fuel-=t.getLightUpkeep()}))"
    )
    if text.count(old_night) != 1:
        raise RuntimeError("Unable to instrument handleNight in the bundled Lux engine")
    text = text.replace(old_night, new_night, 1)

    old_meta = (
        "globalCityIDCount:n.game.globalCityIDCount,"
        "globalUnitIDCount:n.game.globalUnitIDCount}))"
    )
    new_meta = (
        "globalCityIDCount:n.game.globalCityIDCount,"
        "globalUnitIDCount:n.game.globalUnitIDCount,"
        "phase1NightLosses:n.game.__phase1NightLosses||[]}))"
    )
    if text.count(old_meta) != 1:
        raise RuntimeError("Unable to instrument match metadata in the bundled Lux engine")
    text = text.replace(old_meta, new_meta, 1)
    (temporary / "main.js").write_text(text, encoding="utf-8")

    try:
        temporary.replace(target_dir)
    except FileExistsError:
        shutil.rmtree(str(temporary))
    return target_main


class DiagnosticLuxEnv(LuxEnv):
    """LuxEnv variant that reads diagnostics from an instrumented Node engine."""

    def __init__(self, *args: Any, diagnostic_engine: Path, **kwargs: Any) -> None:
        self.diagnostic_engine = diagnostic_engine
        self.last_night_city_losses: List[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _restart_dimension_process(self) -> None:
        if self._dimension_process is not None:
            self._dimension_process.kill()
        if self.run_game_automatically:
            self._dimension_process = Popen(
                ["node", str(self.diagnostic_engine)],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
            )
            self._q = Queue()
            self._t = Thread(target=_enqueue_output, args=(self._dimension_process.stdout, self._q))
            self._t.daemon = True
            self._t.start()

    def _step(self, action: List[List[str]]) -> None:
        state = [{"action": actions} for actions in action]
        self._dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self._dimension_process.stdin.flush()

        agent_1 = json.loads(self._dimension_process.stderr.readline())
        _agent_2 = self._dimension_process.stderr.readline()
        metadata = json.loads(self._dimension_process.stderr.readline())
        self.game_state._update(agent_1)
        status = json.loads(self._dimension_process.stderr.readline())
        self.done = status["status"] == "finished"
        self.last_night_city_losses = metadata.get("phase1NightLosses", [])

        while True:
            try:
                line = self._q.get_nowait()
            except Empty:
                break
            else:
                print(line.decode(), file=sys.stderr, end="")

    def step_action_strings(self, action: List[List[str]]) -> None:
        self._step(action)
        self._update_internal_state()

    def close_engine(self) -> None:
        process = self._dimension_process
        self._dimension_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait()


def _pad_array(value: np.ndarray, board_dims: Tuple[int, int]) -> np.ndarray:
    if value.ndim == 4 and value.shape[2:4] == board_dims:
        return np.pad(
            value,
            ((0, 0), (0, 0), (0, MAX_BOARD_SIZE[0] - board_dims[0]),
             (0, MAX_BOARD_SIZE[1] - board_dims[1])),
        )
    if value.ndim == 5 and value.shape[2:4] == board_dims:
        return np.pad(
            value,
            ((0, 0), (0, 0), (0, MAX_BOARD_SIZE[0] - board_dims[0]),
             (0, MAX_BOARD_SIZE[1] - board_dims[1]), (0, 0)),
        )
    return value


def _stack_nested(values: Sequence[Any]) -> Any:
    if isinstance(values[0], dict):
        return {key: _stack_nested([value[key] for value in values]) for key in values[0]}
    if torch.is_tensor(values[0]):
        return torch.cat(values, dim=0)
    return torch.from_numpy(np.stack(values, axis=0))


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value.to(device=device, non_blocking=True)


def _slice_nested(value: Any, start: int, end: int) -> Any:
    if isinstance(value, dict):
        return {key: _slice_nested(item, start, end) for key, item in value.items()}
    return value[start:end]


@dataclass
class EncodedState:
    obs: Dict[str, torch.Tensor]
    input_mask: torch.Tensor
    available_actions_mask: Dict[str, torch.Tensor]
    game_state: Any

    def model_input(self) -> Dict[str, Any]:
        return {
            "obs": self.obs,
            "info": {
                "input_mask": self.input_mask,
                "available_actions_mask": self.available_actions_mask,
            },
        }


class AgentRuntime:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.device = torch.device(spec.device if torch.cuda.is_available() else "cpu")
        with spec.model_config.open("r", encoding="utf-8") as config_file:
            model_values = yaml.safe_load(config_file)
        # Agent configs written before a model kwarg existed (e.g. the 2021-10
        # agents predate sum_player_embeddings) would crash create_model, so
        # fill those keys with the value the trainer used at the time.
        for key, default in OPTIONAL_MODEL_KEY_DEFAULTS.items():
            model_values.setdefault(key, default)
        self.model_flags = flags_to_namespace(model_values)
        with spec.agent_config.open("r", encoding="utf-8") as config_file:
            agent_values = yaml.safe_load(config_file)
        agent_values.update(spec.overrides)
        self.agent_flags = SimpleNamespace(**agent_values)

        self.obs_space = self.model_flags.obs_space()
        self._obs_wrapper = None
        self.model = create_model(self.model_flags, self.device)
        try:
            checkpoint = torch.load(spec.checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(spec.checkpoint, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.inference_seconds = 0.0
        self.inference_calls = 0
        self.inferred_states = 0

    def _ensure_wrapper(self, env: DiagnosticLuxEnv) -> None:
        if self._obs_wrapper is None:
            self._obs_wrapper = self.obs_space.wrap_env(env)

    def encode(self, env: DiagnosticLuxEnv) -> EncodedState:
        self._ensure_wrapper(env)
        board_dims = (env.game_state.map_width, env.game_state.map_height)
        obs_np = {
            key: _pad_array(value, board_dims)
            for key, value in self._obs_wrapper.observation(env.game_state).items()
        }
        actions_np = {
            key: _pad_array(value, board_dims)
            for key, value in env.info["available_actions_mask"].items()
        }
        input_mask = np.zeros((1, *MAX_BOARD_SIZE), dtype=bool)
        input_mask[:, :board_dims[0], :board_dims[1]] = True
        return EncodedState(
            obs={key: torch.from_numpy(value).unsqueeze(0) for key, value in obs_np.items()},
            input_mask=torch.from_numpy(input_mask).unsqueeze(0),
            available_actions_mask={
                key: torch.from_numpy(value).unsqueeze(0) for key, value in actions_np.items()
            },
            game_state=env.game_state,
        )

    def _augmenters(self, game_state: Any) -> List[data_augmentation.DataAugmenter]:
        augmenters = []
        for factory_name in getattr(self.agent_flags, "data_augmentations", []):
            factory = data_augmentation.__dict__.get(factory_name)
            if factory is None:
                raise ValueError("Unknown data augmentation: {}".format(factory_name))
            augmenter = factory(game_state=game_state)
            if not isinstance(augmenter, data_augmentation.DataAugmenter):
                raise TypeError("{} did not create a DataAugmenter".format(factory_name))
            augmenters.append(augmenter)
        return augmenters

    @staticmethod
    def _apply_augmentation(
        model_input: Dict[str, Any], augmenter: data_augmentation.DataAugmenter
    ) -> Dict[str, Any]:
        return {
            "obs": augmenter.apply(model_input["obs"], inverse=False, is_policy=False),
            "info": {
                "input_mask": augmenter.apply(
                    {"input_mask": model_input["info"]["input_mask"].unsqueeze(1)},
                    inverse=False,
                    is_policy=False,
                )["input_mask"].squeeze(1),
                "available_actions_mask": augmenter.apply(
                    model_input["info"]["available_actions_mask"],
                    inverse=False,
                    is_policy=True,
                ),
            },
        }

    def infer(self, states: Sequence[EncodedState]) -> List[Dict[str, Any]]:
        if not states:
            return []
        variants: List[Dict[str, Any]] = []
        groups: List[Tuple[int, List[data_augmentation.DataAugmenter]]] = []
        for state in states:
            model_input = state.model_input()
            augmenters = self._augmenters(state.game_state)
            start = len(variants)
            variants.append(model_input)
            variants.extend(self._apply_augmentation(model_input, aug) for aug in augmenters)
            groups.append((start, augmenters))

        batched = _move_nested(_stack_nested(variants), self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.select_best_actions(batched)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        self.inference_seconds += elapsed
        self.inference_calls += 1
        self.inferred_states += len(states)

        results = []
        for start, augmenters in groups:
            policies = [
                {key: value[start:start + 1].cpu() for key, value in output["policy_logits"].items()}
            ]
            for offset, augmenter in enumerate(augmenters, 1):
                policy = {
                    key: value[start + offset:start + offset + 1].cpu()
                    for key, value in output["policy_logits"].items()
                }
                policies.append(augmenter.apply(policy, inverse=True, is_policy=True))
            logits = {
                key: torch.cat([policy[key] for policy in policies], dim=0).mean(dim=0, keepdim=True)
                for key in policies[0]
            }
            actions = {
                key: DictActor.logits_to_actions(
                    torch.flatten(value, start_dim=0, end_dim=-2),
                    sample=False,
                    actions_per_square=None,
                ).view(*value.shape[:-1], -1)
                for key, value in logits.items()
            }
            baseline = output["baseline"][start:start + len(augmenters) + 1].mean(
                dim=0, keepdim=True
            ).cpu()
            results.append({"policy_logits": logits, "actions": actions, "baseline": baseline})
        return results


def _pos_to_loc(position: Tuple[int, int]) -> int:
    return position[0] * MAX_BOARD_SIZE[1] + position[1]


def _resolve_player_actions(
    env: DiagnosticLuxEnv,
    player_id: int,
    agent_output: Dict[str, Any],
    agent_flags: SimpleNamespace,
) -> List[str]:
    if not getattr(agent_flags, "use_collision_detection", True):
        actions, _ = env.process_actions({
            key: value.squeeze(0).numpy() for key, value in agent_output["actions"].items()
        })
        return actions[player_id]

    game_state = env.game_state
    player = game_state.players[player_id]
    my_city_tile_mat = np.zeros(MAX_BOARD_SIZE, dtype=bool)
    actionable_city_tiles: Dict[int, Any] = {}
    actionable_workers: Dict[int, List[Any]] = {}
    actionable_carts: Dict[int, List[Any]] = {}
    for unit in player.units:
        if not unit.can_act():
            continue
        target = actionable_workers if unit.is_worker() else actionable_carts
        target.setdefault(_pos_to_loc(unit.pos.astuple()), []).append(unit)
    for city_tile in player.city_tiles:
        my_city_tile_mat[city_tile.pos.x, city_tile.pos.y] = True
        if city_tile.can_act():
            actionable_city_tiles[_pos_to_loc(city_tile.pos.astuple())] = city_tile

    flat_log_probs = {
        key: torch.flatten(
            F.log_softmax(value.squeeze(0).squeeze(0), dim=-1),
            start_dim=-3,
            end_dim=-2,
        ).clone()
        for key, value in agent_output["policy_logits"].items()
    }
    my_flat_log_probs = {key: value[player_id] for key, value in flat_log_probs.items()}
    my_flat_actions = {
        key: torch.flatten(
            value.squeeze(0).squeeze(0)[player_id], start_dim=-3, end_dim=-2
        )
        for key, value in agent_output["actions"].items()
    }

    units_to_build = max(player.city_tile_count - len(player.units), 0)
    research_remaining = max(MAX_RESEARCH - player.research_points, 0)
    city_scores = my_flat_log_probs["city_tile"].max(dim=-1)[0]
    city_order = torch.argsort(city_scores, descending=True)
    city_mask = torch.zeros_like(city_scores, dtype=torch.bool)
    if actionable_city_tiles:
        city_mask[list(actionable_city_tiles)] = True
    city_locations = city_order[city_mask[city_order]]
    for location in city_locations.tolist():
        for action_tensor in my_flat_actions["city_tile"][location]:
            action = int(action_tensor.item())
            meaning = ACTION_MEANINGS["city_tile"][action]
            illegal = False
            if meaning == "BUILD_CART" and not getattr(agent_flags, "can_build_carts", False):
                illegal = True
            elif meaning.startswith("BUILD_"):
                if units_to_build > 0:
                    units_to_build -= 1
                else:
                    illegal = True
            elif meaning == "RESEARCH":
                if research_remaining > 0:
                    research_remaining -= 1
                else:
                    illegal = True
            elif (
                meaning == "NO-OP"
                and game_state.turn >= DN_CYCLE_LEN
                and research_remaining > 0
                and getattr(agent_flags, "must_research", False)
            ):
                illegal = True
            if game_state.turn >= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] - 1:
                illegal = meaning != "BUILD_CART"
            if illegal:
                my_flat_log_probs["city_tile"][location, action] = float("-inf")
            else:
                break

    occupied = np.zeros(MAX_BOARD_SIZE, dtype=bool)
    max_locations = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]
    combined = torch.cat(
        [
            my_flat_log_probs["worker"].max(dim=-1)[0],
            my_flat_log_probs["cart"].max(dim=-1)[0],
        ],
        dim=-1,
    )
    actionable_locations = list(actionable_workers)
    actionable_locations.extend(max_locations + location for location in actionable_carts)
    unit_order = torch.argsort(combined, descending=True)
    unit_mask = torch.zeros_like(combined, dtype=torch.bool)
    if actionable_locations:
        unit_mask[actionable_locations] = True
    unit_locations = unit_order[unit_mask[unit_order]]
    for location in unit_locations.tolist():
        if location >= max_locations:
            unit_type = "cart"
            actionable = actionable_carts
        else:
            unit_type = "worker"
            actionable = actionable_workers
        location %= max_locations
        units = actionable[location]
        acted = 0
        for action_tensor in my_flat_actions[unit_type][location]:
            action = int(action_tensor.item())
            meaning = ACTION_MEANINGS[unit_type][action]
            if meaning.startswith("MOVE_"):
                new_position = units[acted].pos.translate(meaning.split("_")[1], 1)
            else:
                new_position = units[acted].pos
            illegal = (
                new_position.x < 0
                or new_position.x >= game_state.map_width
                or new_position.y < 0
                or new_position.y >= game_state.map_height
                or (
                    occupied[new_position.x, new_position.y]
                    and not my_city_tile_mat[new_position.x, new_position.y]
                )
            )
            if illegal:
                my_flat_log_probs[unit_type][location, action] = float("-inf")
            else:
                occupied[new_position.x, new_position.y] = True
                acted += 1
            if acted >= len(units):
                break

    action_tensors = {
        key: value.view(1, *value.shape[:-2], *MAX_BOARD_SIZE, -1).argsort(
            dim=-1, descending=True
        )
        for key, value in flat_log_probs.items()
    }
    actions, _ = env.process_actions({key: value.numpy() for key, value in action_tensors.items()})
    return actions[player_id]


def _count_units(game_state: Any) -> Tuple[int, int]:
    return tuple(len(player.units) for player in game_state.players)


def _count_city_tiles(game_state: Any) -> Tuple[int, int]:
    return tuple(player.city_tile_count for player in game_state.players)


def _unit_snapshot(game_state: Any) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"turn": int(game_state.turn), "players": []}
    for player in game_state.players:
        snapshot["players"].append({
            "workers": sum(unit.is_worker() for unit in player.units),
            "carts": sum(unit.is_cart() for unit in player.units),
            "total": len(player.units),
            "city_tiles": player.city_tile_count,
            "research_points": player.research_points,
        })
    return snapshot


class GameSlot:
    def __init__(
        self,
        request: MatchRequest,
        output_dir: Path,
        engine_path: Path,
        configuration: Mapping[str, Any],
    ) -> None:
        self.request = request
        self.output_dir = output_dir
        config = copy.deepcopy(dict(configuration))
        config.update({
            "width": request.map_size,
            "height": request.map_size,
            "seed": request.seed,
            "loglevel": 0,
        })
        model_values = yaml.safe_load(request.agent_0.model_config.read_text(encoding="utf-8"))
        model_flags = flags_to_namespace(model_values)
        self.env = DiagnosticLuxEnv(
            act_space=model_flags.act_space(),
            obs_space=model_flags.obs_space(),
            configuration=config,
            seed=request.seed,
            run_game_automatically=True,
            diagnostic_engine=engine_path,
        )
        self.env.reset()
        if self.env.get_seed() != request.seed:
            raise RuntimeError("Engine seed mismatch: {} != {}".format(self.env.get_seed(), request.seed))
        self.started = time.monotonic()
        self.commands: List[List[Dict[str, Any]]] = []
        self.first_city_loss_turn: List[Optional[int]] = [None, None]
        self.night_city_losses: List[Dict[str, Any]] = []
        self.research_completion_turn: List[Optional[int]] = [None, None]
        self.coal_completion_turn: List[Optional[int]] = [None, None]
        self.unit_counts_over_time = [_unit_snapshot(self.env.game_state)]
        self.previous_city_positions = self._city_positions()
        self.previous_research = [player.research_points for player in self.env.game_state.players]

    def _city_positions(self) -> List[set]:
        return [
            {(tile.pos.x, tile.pos.y) for tile in player.city_tiles}
            for player in self.env.game_state.players
        ]

    def step(self, actions: List[List[str]]) -> None:
        turn = int(self.env.game_state.turn)
        self.commands.append([
            {"command": command, "agentID": player_id}
            for player_id, player_actions in enumerate(actions)
            for command in player_actions
        ])
        self.env.step_action_strings(actions)

        current_positions = self._city_positions()
        for player_id in (0, 1):
            if (
                self.first_city_loss_turn[player_id] is None
                and self.previous_city_positions[player_id] - current_positions[player_id]
            ):
                self.first_city_loss_turn[player_id] = turn
        self.previous_city_positions = current_positions

        for event in self.env.last_night_city_losses:
            event = dict(event)
            event["turn"] = int(event["turn"])
            event["team"] = int(event["team"])
            event["fuel"] = float(event["fuel"])
            event["upkeep"] = float(event["upkeep"])
            event["shortfall"] = float(event["shortfall"])
            self.night_city_losses.append(event)

        for player_id, player in enumerate(self.env.game_state.players):
            if self.coal_completion_turn[player_id] is None and player.research_points >= 50:
                self.coal_completion_turn[player_id] = turn
            if self.research_completion_turn[player_id] is None and player.research_points >= MAX_RESEARCH:
                self.research_completion_turn[player_id] = turn
            self.previous_research[player_id] = player.research_points
        self.unit_counts_over_time.append(_unit_snapshot(self.env.game_state))

    def _winner(self) -> Optional[int]:
        city_tiles = _count_city_tiles(self.env.game_state)
        if city_tiles[0] != city_tiles[1]:
            return 0 if city_tiles[0] > city_tiles[1] else 1
        units = _count_units(self.env.game_state)
        if units[0] != units[1]:
            return 0 if units[0] > units[1] else 1
        return None

    def finish(self, cache_key: str) -> MatchResult:
        winner = self._winner()
        diagnostics_path = self.output_dir / "diagnostics" / (self.request.match_id + ".json")
        diagnostics = {
            "match_id": self.request.match_id,
            "seed": self.request.seed,
            "map_size": self.request.map_size,
            "candidate_seat": self.request.candidate_seat,
            "first_city_loss_turn": self.first_city_loss_turn,
            "night_city_losses": self.night_city_losses,
            "coal_research_completion_turn": self.coal_completion_turn,
            "uranium_research_completion_turn": self.research_completion_turn,
            "unit_counts_over_time": self.unit_counts_over_time,
        }
        _json_dump(diagnostics_path, diagnostics)

        replay_path: Optional[Path] = None
        if winner is not None and winner != self.request.candidate_seat:
            replay_path = self.output_dir / "replays" / (self.request.match_id + ".json")
            ranks = [
                {"rank": 1 if player_id == winner else 2, "agentID": player_id}
                for player_id in (0, 1)
            ]
            replay = {
                "seed": self.request.seed,
                "allCommands": self.commands,
                "mapType": "random",
                "width": self.request.map_size,
                "height": self.request.map_size,
                "teamDetails": [
                    {"name": self.request.agent_0.name, "tournamentID": ""},
                    {"name": self.request.agent_1.name, "tournamentID": ""},
                ],
                "version": "3.1.0",
                "results": {"ranks": ranks, "replayFile": str(replay_path)},
            }
            _json_dump(replay_path, replay)

        return MatchResult(
            match_id=self.request.match_id,
            seed=self.request.seed,
            map_size=self.request.map_size,
            agent_0=self.request.agent_0.name,
            agent_1=self.request.agent_1.name,
            agent_0_hash=self.request.agent_0.content_hash,
            agent_1_hash=self.request.agent_1.content_hash,
            candidate_seat=self.request.candidate_seat,
            winner=winner,
            final_city_tiles=_count_city_tiles(self.env.game_state),
            final_units=_count_units(self.env.game_state),
            ending_turn=int(self.env.game_state.turn),
            wall_seconds=time.monotonic() - self.started,
            replay_path=str(replay_path) if replay_path else None,
            diagnostics_path=str(diagnostics_path),
            cache_key=cache_key,
        )

    def close(self) -> None:
        self.env.close_engine()


def _cache_key(request: MatchRequest) -> str:
    value = {
        "version": HARNESS_VERSION,
        "seed": request.seed,
        "map_size": request.map_size,
        "agent_0_hash": request.agent_0.content_hash,
        "agent_1_hash": request.agent_1.content_hash,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


class BatchedMatchRunner:
    def __init__(
        self,
        output_dir: Union[str, Path],
        parallel_games: int = 8,
        use_cache: bool = True,
    ) -> None:
        if parallel_games < 1:
            raise ValueError("parallel_games must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_games = parallel_games
        self.use_cache = use_cache
        self.engine_path = _prepare_diagnostic_engine(self.output_dir)
        from kaggle_environments import make

        self.configuration = dict(make("lux_ai_2021").configuration)
        self.runtimes: Dict[str, AgentRuntime] = {}
        self.stats = RunnerStats()

    def _runtime(self, spec: AgentSpec) -> AgentRuntime:
        runtime = self.runtimes.get(spec.content_hash)
        if runtime is None:
            runtime = AgentRuntime(spec)
            self.runtimes[spec.content_hash] = runtime
        return runtime

    def _load_cached(self, request: MatchRequest) -> Optional[MatchResult]:
        if not self.use_cache:
            return None
        key = _cache_key(request)
        path = self.output_dir / "cache" / (key + ".json")
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as cache_file:
            result = MatchResult.from_dict(json.load(cache_file))
        # Candidate seat and match IDs are report labels, not simulation inputs.
        result.match_id = request.match_id
        result.candidate_seat = request.candidate_seat
        self.stats.cache_hits += 1
        return result

    def _save_cached(self, result: MatchResult) -> None:
        path = self.output_dir / "cache" / (result.cache_key + ".json")
        _json_dump(path, result.to_dict())

    def run(self, requests: Sequence[MatchRequest]) -> List[MatchResult]:
        started = time.monotonic()
        results: Dict[str, MatchResult] = {}
        pending: List[MatchRequest] = []
        for request in requests:
            cached = self._load_cached(request)
            if cached is None:
                pending.append(request)
            else:
                results[request.match_id] = cached

        slots: List[GameSlot] = []
        executor = ThreadPoolExecutor(max_workers=self.parallel_games)
        try:
            while pending or slots:
                while pending and len(slots) < self.parallel_games:
                    request = pending.pop(0)
                    slots.append(GameSlot(request, self.output_dir, self.engine_path, self.configuration))

                needs: Dict[str, List[GameSlot]] = {}
                specs: Dict[str, AgentSpec] = {}
                for slot in slots:
                    for spec in (slot.request.agent_0, slot.request.agent_1):
                        specs[spec.content_hash] = spec
                        if slot not in needs.setdefault(spec.content_hash, []):
                            needs[spec.content_hash].append(slot)

                outputs: Dict[Tuple[str, int], Dict[str, Any]] = {}
                for agent_hash, needed_slots in needs.items():
                    runtime = self._runtime(specs[agent_hash])
                    encoded = [runtime.encode(slot.env) for slot in needed_slots]
                    inferred = runtime.infer(encoded)
                    for slot, output in zip(needed_slots, inferred):
                        outputs[(agent_hash, id(slot))] = output

                actions_by_slot = []
                for slot in slots:
                    output_0 = outputs[(slot.request.agent_0.content_hash, id(slot))]
                    output_1 = outputs[(slot.request.agent_1.content_hash, id(slot))]
                    actions_by_slot.append([
                        _resolve_player_actions(
                            slot.env, 0, output_0, self._runtime(slot.request.agent_0).agent_flags
                        ),
                        _resolve_player_actions(
                            slot.env, 1, output_1, self._runtime(slot.request.agent_1).agent_flags
                        ),
                    ])

                engine_started = time.monotonic()
                futures = [
                    executor.submit(slot.step, actions)
                    for slot, actions in zip(slots, actions_by_slot)
                ]
                for future in futures:
                    future.result()
                self.stats.engine_seconds += time.monotonic() - engine_started

                active = []
                for slot in slots:
                    if slot.env.done:
                        key = _cache_key(slot.request)
                        result = slot.finish(key)
                        results[slot.request.match_id] = result
                        self._save_cached(result)
                        self.stats.total_turns += result.ending_turn
                        slot.close()
                    else:
                        active.append(slot)
                slots = active
        finally:
            executor.shutdown(wait=True)
            for slot in slots:
                slot.close()

        self.stats.wall_seconds = time.monotonic() - started
        self.stats.model_seconds = sum(runtime.inference_seconds for runtime in self.runtimes.values())
        self.stats.inference_calls = sum(runtime.inference_calls for runtime in self.runtimes.values())
        self.stats.inferred_states = sum(runtime.inferred_states for runtime in self.runtimes.values())
        return [results[request.match_id] for request in requests]


def build_paired_schedule(
    candidate: AgentSpec,
    baseline: AgentSpec,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    map_sizes: Sequence[int] = DEFAULT_MAP_SIZES,
) -> List[MatchRequest]:
    requests = []
    for map_size in map_sizes:
        for seed in seeds:
            prefix = "m{:02d}_seed{:09d}".format(map_size, seed)
            requests.append(MatchRequest(
                agent_0=candidate,
                agent_1=baseline,
                seed=seed,
                map_size=map_size,
                candidate_seat=0,
                match_id=prefix + "_candidate_p0",
            ))
            requests.append(MatchRequest(
                agent_0=baseline,
                agent_1=candidate,
                seed=seed,
                map_size=map_size,
                candidate_seat=1,
                match_id=prefix + "_candidate_p1",
            ))
    return requests


def wilson_interval(successes: float, games: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if games <= 0:
        return float("nan"), float("nan")
    proportion = successes / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_results(results: Sequence[MatchResult], runner_stats: RunnerStats) -> Dict[str, Any]:
    def group(values: Sequence[MatchResult]) -> Dict[str, Any]:
        games = len(values)
        wins = sum(result.candidate_score == 1.0 for result in values)
        losses = sum(result.candidate_score == 0.0 for result in values)
        ties = games - wins - losses
        successes = sum(result.candidate_score for result in values)
        low, high = wilson_interval(successes, games)
        margins = [result.candidate_city_margin for result in values]
        turns = sum(result.ending_turn for result in values)
        return {
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": successes / games if games else float("nan"),
            "raw_decisive_win_rate": wins / games if games else float("nan"),
            "ci95": [low, high],
            "mean_city_tile_margin": float(np.mean(margins)) if margins else float("nan"),
            "median_city_tile_margin": float(np.median(margins)) if margins else float("nan"),
            "mean_match_seconds_per_step": (
                sum(result.wall_seconds for result in values) / max(turns, 1)
            ),
        }

    return {
        "primary_metric_note": "win_rate scores a tie as one half; ci95 is a Wilson score interval",
        "aggregate": group(results),
        "by_seat": {
            "player_0": group([result for result in results if result.candidate_seat == 0]),
            "player_1": group([result for result in results if result.candidate_seat == 1]),
        },
        "by_map_size": {
            str(map_size): group([result for result in results if result.map_size == map_size])
            for map_size in sorted({result.map_size for result in results})
        },
        "performance": runner_stats.to_dict(),
    }


def paired_repeat_mismatches(results: Sequence[MatchResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[MatchResult]] = {}
    for result in results:
        grouped.setdefault((result.seed, result.map_size), []).append(result)
    mismatches = []
    for (seed, map_size), pair in grouped.items():
        if len(pair) != 2:
            mismatches.append({"seed": seed, "map_size": map_size, "reason": "expected two seats"})
            continue
        physical = {
            (result.winner, result.final_city_tiles, result.final_units, result.ending_turn)
            for result in pair
        }
        if len(physical) != 1:
            mismatches.append({
                "seed": seed,
                "map_size": map_size,
                "reason": "identical-agent repeat was nondeterministic",
                "results": [result.to_dict() for result in pair],
            })
    return mismatches


def write_report(
    output_dir: Union[str, Path],
    results: Sequence[MatchResult],
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    output_dir = Path(output_dir)
    _json_dump(output_dir / "summary.json", {"metadata": dict(metadata), "summary": dict(summary)})
    fieldnames = list(results[0].to_dict()) if results else []
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)

    aggregate = summary["aggregate"]
    lines = [
        "# Lux AI Phase 1 evaluation",
        "",
        "- Games: {}".format(aggregate["games"]),
        "- Win rate (ties = 0.5): {:.2%} (95% CI {:.2%} to {:.2%})".format(
            aggregate["win_rate"], *aggregate["ci95"]
        ),
        "- Wins/losses/ties: {}/{}/{}".format(
            aggregate["wins"], aggregate["losses"], aggregate["ties"]
        ),
        "- Mean final city-tile margin: {:.3f}".format(aggregate["mean_city_tile_margin"]),
        "- Throughput: {:.3f} game steps/s ({:.6f} wall s/game step)".format(
            summary["performance"]["steps_per_wall_second"],
            summary["performance"]["throughput_seconds_per_step"],
        ),
        "",
        "## Seat split",
        "",
    ]
    for seat, values in summary["by_seat"].items():
        lines.append("- {}: {:.2%} (95% CI {:.2%} to {:.2%}), margin {:.3f}".format(
            seat, values["win_rate"], *values["ci95"], values["mean_city_tile_margin"]
        ))
    lines.extend(["", "## Map-size split", ""])
    for map_size, values in summary["by_map_size"].items():
        lines.append("- {}: {:.2%} (95% CI {:.2%} to {:.2%}), margin {:.3f}".format(
            map_size, values["win_rate"], *values["ci95"], values["mean_city_tile_margin"]
        ))
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def play_match(
    agent_a: Union[AgentSpec, str, Path],
    agent_b: Union[AgentSpec, str, Path],
    seed: int,
    map_size: int,
    output_dir: Union[str, Path] = "evaluation_runs/play_match",
    device: str = "cuda:0",
) -> MatchResult:
    """Play one physical match with agent_a in player seat 0 and agent_b in seat 1."""
    if not isinstance(agent_a, AgentSpec):
        agent_a = AgentSpec.from_directory(agent_a, name=Path(agent_a).name, device=device)
    if not isinstance(agent_b, AgentSpec):
        agent_b = AgentSpec.from_directory(agent_b, name=Path(agent_b).name, device=device)
    request = MatchRequest(
        agent_0=agent_a,
        agent_1=agent_b,
        seed=seed,
        map_size=map_size,
        candidate_seat=0,
        match_id="play_match_m{}_seed{}".format(map_size, seed),
    )
    runner = BatchedMatchRunner(output_dir=output_dir, parallel_games=1, use_cache=False)
    return runner.run([request])[0]
