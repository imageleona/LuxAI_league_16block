import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union


def replace_with_retry(tmp_path: Union[str, Path], path: Union[str, Path],
                       attempts: int = 5, backoff: float = 0.05) -> bool:
    """
    os.replace, retried.

    On Windows os.replace fails with PermissionError if any other process holds
    the destination open. POSIX rename has no such restriction, so the naive
    version works everywhere except here, then kills a long run hours in once the
    race finally lands. Returns True if the rename succeeded.

    Callers decide what a failure means: publishing the league state is idempotent
    and can be skipped, but a checkpoint that did not land must not be treated as
    written - see LeagueManager.write_agent_dir.
    """
    for attempt in range(attempts):
        try:
            os.replace(str(tmp_path), str(path))
            return True
        except PermissionError:
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    return False


def write_state_atomic(path: Union[str, Path], state: Dict,
                       attempts: int = 5, backoff: float = 0.05) -> bool:
    """
    Atomically publish the league state (write tmp, then os.replace).

    On Windows os.replace fails with PermissionError if any other process holds
    the destination open - and the actors read this file. POSIX rename has no
    such restriction, so the naive version works everywhere except here, then
    kills a long run hours in once the race finally lands.

    Publishing is idempotent and happens every few seconds, so a lost write costs
    nothing: retry briefly, and if it still fails give up and let the next publish
    carry the same information. Returns True if the state reached disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    if replace_with_retry(tmp_path, path, attempts=attempts, backoff=backoff):
        return True
    logging.warning(
        "League: could not publish %s (a reader held it open on every attempt); "
        "skipping this update, the next publish will carry the same state.", path)
    return False


def read_state(path: Union[str, Path]) -> Optional[Dict]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A torn read can only happen if os.replace is not atomic on this
        # filesystem; the caller just keeps its previous state and retries later.
        return None


def read_state_if_newer(path: Union[str, Path], version: int) -> Optional[Dict]:
    state = read_state(path)
    if state is None or state.get("version", -1) <= version:
        return None
    return state
