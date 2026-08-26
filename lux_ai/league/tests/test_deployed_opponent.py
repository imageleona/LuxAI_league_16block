import numpy as np

from lux_ai.lux_gym.lux_env import LuxEnv


def test_lux_env_applies_one_shot_player_command_override():
    env = LuxEnv.__new__(LuxEnv)
    env.run_game_automatically = True
    env._player_action_overrides = {}
    env.info = {}
    captured = []
    actions_taken = {
        "worker": np.ones((1, 2, 1, 1, 1), dtype=bool),
        "city_tile": np.ones((1, 2, 1, 1, 1), dtype=bool),
    }
    env.process_actions = lambda action: (["raw-0", "raw-1"], actions_taken)
    env._step = lambda actions: captured.append(actions)
    env._update_internal_state = lambda: None
    env.get_obs_reward_done_info = lambda: "result"

    env.override_player_actions(1, ["m u_1 n", "r 2 3"])
    result = LuxEnv.step(env, {})

    assert result == "result"
    assert captured == [["raw-0", ["m u_1 n", "r 2 3"]]]
    assert not actions_taken["worker"][:, 1].any()
    assert not actions_taken["city_tile"][:, 1].any()
    assert env._player_action_overrides == {}


def test_player_action_override_rejects_unknown_seat():
    env = LuxEnv.__new__(LuxEnv)
    env._player_action_overrides = {}

    try:
        env.override_player_actions(2, [])
    except ValueError as exc:
        assert "player must be 0 or 1" in str(exc)
    else:
        raise AssertionError("invalid player seat was accepted")
