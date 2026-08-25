from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

ANCHOR = "anchor"
EXTERNAL = "external"
SNAPSHOT = "snapshot"
MEMBER_KINDS = (ANCHOR, EXTERNAL, SNAPSHOT)


@dataclass
class PoolMember:
    member_id: str
    # Absolute path of the agent directory (the dir containing lux_ai/rl_agent/
    # or, for flat layouts, config.yaml + weights directly).
    dir_path: str
    # Absolute path of the model config yaml and the weights file.
    config_path: str
    checkpoint_path: str
    kind: str = SNAPSHOT
    # Which prefix the opponent model must read from the training env's obs dict:
    # "student_", "teacher_" (MultiObs envs) or "" (single obs space).
    obs_space_prefix: str = ""

    def to_dict(self) -> Dict:
        return {
            "member_id": self.member_id,
            "dir_path": self.dir_path,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "kind": self.kind,
            "obs_space_prefix": self.obs_space_prefix,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PoolMember":
        return cls(**d)


class OpponentPool:
    """
    Bounded pool of frozen opponents. Anchors are permanent; the remaining
    slots are filled by reservoir sampling over the stream of offered members
    (so the pool keeps a uniform sample of history instead of collapsing to
    the most recent snapshots).
    """

    def __init__(self, pool_size: int, rng: Optional[np.random.RandomState] = None):
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, was {pool_size}")
        self.pool_size = pool_size
        self.rng = rng if rng is not None else np.random.RandomState()
        self._anchors: List[PoolMember] = []
        self._others: List[PoolMember] = []
        # Total number of non-anchor members ever offered (reservoir stream count).
        self.n_offered = 0

    @property
    def members(self) -> List[PoolMember]:
        return list(self._anchors) + list(self._others)

    @property
    def member_ids(self) -> List[str]:
        return [m.member_id for m in self.members]

    def __len__(self) -> int:
        return len(self._anchors) + len(self._others)

    def __contains__(self, member_id: str) -> bool:
        return member_id in self.member_ids

    def get(self, member_id: str) -> PoolMember:
        for member in self.members:
            if member.member_id == member_id:
                return member
        raise KeyError(member_id)

    def add_anchor(self, member: PoolMember) -> None:
        if member.member_id in self:
            raise ValueError(f"Duplicate member_id: {member.member_id}")
        if len(self._anchors) >= self.pool_size:
            raise ValueError("Cannot add anchor: pool is full of anchors")
        member.kind = ANCHOR
        self._anchors.append(member)

    def offer(self, member: PoolMember) -> Tuple[bool, Optional[PoolMember]]:
        """
        Offer a non-anchor member to the reservoir.
        Returns (accepted, evicted_member).
        """
        if member.member_id in self:
            raise ValueError(f"Duplicate member_id: {member.member_id}")
        capacity = self.pool_size - len(self._anchors)
        self.n_offered += 1
        if capacity <= 0:
            return False, None
        if len(self._others) < capacity:
            self._others.append(member)
            return True, None
        # Reservoir sampling (algorithm R): keep the newcomer with prob capacity / n_offered,
        # replacing a uniformly random non-anchor member.
        if self.rng.rand() < capacity / self.n_offered:
            evict_idx = self.rng.randint(len(self._others))
            evicted = self._others[evict_idx]
            self._others[evict_idx] = member
            return True, evicted
        return False, None

    def replace(self, member: PoolMember, evict_id: Optional[str] = None) -> Optional[PoolMember]:
        """
        Admit `member` unconditionally, evicting a non-anchor to make room.

        Unlike offer(), which is a reservoir coin flip, this always succeeds - it is
        for members that earned their place (see winrate_admit_threshold). Anchors are
        never evicted. `evict_id` names the victim; without it a uniformly random
        non-anchor is dropped. Returns the evicted member, or None if there was room.
        """
        if member.member_id in self:
            raise ValueError(f"Duplicate member_id: {member.member_id}")
        capacity = self.pool_size - len(self._anchors)
        if capacity <= 0:
            return None
        self.n_offered += 1
        if len(self._others) < capacity:
            self._others.append(member)
            return None
        if evict_id is not None:
            idx = next((i for i, m in enumerate(self._others) if m.member_id == evict_id), None)
            if idx is None:
                raise KeyError(f"{evict_id} is not an evictable pool member")
        else:
            idx = self.rng.randint(len(self._others))
        evicted = self._others[idx]
        self._others[idx] = member
        return evicted

    def to_dict(self) -> Dict:
        return {
            "pool_size": self.pool_size,
            "n_offered": self.n_offered,
            "anchors": [m.to_dict() for m in self._anchors],
            "others": [m.to_dict() for m in self._others],
        }

    @classmethod
    def from_dict(cls, d: Dict, rng: Optional[np.random.RandomState] = None) -> "OpponentPool":
        pool = cls(d["pool_size"], rng=rng)
        pool.n_offered = d["n_offered"]
        pool._anchors = [PoolMember.from_dict(m) for m in d["anchors"]]
        pool._others = [PoolMember.from_dict(m) for m in d["others"]]
        return pool
