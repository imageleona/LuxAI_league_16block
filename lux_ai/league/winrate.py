from typing import Dict


class WinRateTracker:
    """
    One exponential moving average of the current agent's win rate per pool
    member, updated from training games only.
    outcome: 1.0 win, 0.0 loss, 0.5 draw (from the learner's perspective).
    """

    def __init__(self, ema_lambda: float = 0.1, init: float = 0.5):
        if not 0. < ema_lambda <= 1.:
            raise ValueError(f"ema_lambda must be in (0, 1], was {ema_lambda}")
        self.ema_lambda = ema_lambda
        self.init = init
        self._winrates: Dict[str, float] = {}
        self._games: Dict[str, int] = {}

    def add_member(self, member_id: str) -> None:
        if member_id not in self._winrates:
            self._winrates[member_id] = self.init
            self._games[member_id] = 0

    def remove_member(self, member_id: str) -> None:
        self._winrates.pop(member_id, None)
        self._games.pop(member_id, None)

    def update(self, member_id: str, outcome: float) -> None:
        if not 0. <= outcome <= 1.:
            raise ValueError(f"outcome must be in [0, 1], was {outcome}")
        if member_id not in self._winrates:
            # Outcome for a member that was evicted between game start and game
            # end (or unknown id) - drop it silently, the estimate is gone anyway.
            return
        self._winrates[member_id] = (
            (1. - self.ema_lambda) * self._winrates[member_id] + self.ema_lambda * outcome
        )
        self._games[member_id] += 1

    def winrate(self, member_id: str) -> float:
        return self._winrates[member_id]

    def games(self, member_id: str) -> int:
        return self._games[member_id]

    def to_dict(self) -> Dict:
        return {"winrates": dict(self._winrates), "games": dict(self._games)}

    @classmethod
    def from_dict(cls, d: Dict, ema_lambda: float = 0.1, init: float = 0.5) -> "WinRateTracker":
        tracker = cls(ema_lambda=ema_lambda, init=init)
        tracker._winrates = dict(d["winrates"])
        tracker._games = {key: int(val) for key, val in d["games"].items()}
        return tracker
