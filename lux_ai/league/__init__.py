"""
PFSP opponent league.

This package must not import anything from lux_ai.torchbeast (or any other
learner code). The training loop talks to the league through exactly three
operations:
    - ActorLeagueClient.assign(...)          (select_opponent, per episode)
    - LeagueManager.report_outcome(...)      (per episode)
    - LeagueManager.snapshot(...)            (every N learner updates)

NB: lux_ai.league.actor_client is intentionally not imported here - it pulls
in lux_ai.nns (torch models) and is only needed inside actor processes.
"""
from .flags import LeagueFlags
from .pool import OpponentPool, PoolMember
from .winrate import WinRateTracker
from .sampler import PFSPSampler, PRIORITY_FNS
