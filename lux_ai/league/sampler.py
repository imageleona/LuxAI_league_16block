from typing import Callable, Dict, List


def _priority_pfsp(winrate: float, alpha: float) -> float:
    return (1. - winrate) ** alpha


def _priority_variance(winrate: float, alpha: float) -> float:
    return winrate * (1. - winrate)


def _priority_uniform(winrate: float, alpha: float) -> float:
    return 1.


PRIORITY_FNS: Dict[str, Callable[[float, float], float]] = {
    "pfsp": _priority_pfsp,
    "variance": _priority_variance,
    "uniform": _priority_uniform,
}


class PFSPSampler:
    """
    winrate[i] -> priority fn -> weight[i] -> normalize -> prob[i],
    mixed with a uniform floor so no member stops being sampled entirely.
    """

    def __init__(self, priority: str = "pfsp", alpha: float = 0.5, eps_uniform: float = 0.1):
        if priority not in PRIORITY_FNS:
            raise ValueError(f"priority must be one of {sorted(PRIORITY_FNS)}, was {priority}")
        if not 0. <= eps_uniform <= 1.:
            raise ValueError(f"eps_uniform must be in [0, 1], was {eps_uniform}")
        self.priority = priority
        self.priority_fn = PRIORITY_FNS[priority]
        self.alpha = alpha
        self.eps_uniform = eps_uniform

    def probs(self, winrates: List[float]) -> List[float]:
        n = len(winrates)
        if n == 0:
            return []
        weights = [self.priority_fn(wr, self.alpha) for wr in winrates]
        total = sum(weights)
        if total <= 0.:
            normalized = [1. / n] * n
        else:
            normalized = [w / total for w in weights]
        uniform = 1. / n
        return [(1. - self.eps_uniform) * p + self.eps_uniform * uniform for p in normalized]
