import torch
from typing import Optional

from . import act_spaces, obs_spaces, reward_spaces, multi_subtask
from .lux_env import LuxEnv
from .wrappers import RewardSpaceWrapper, PadFixedShapeEnv, LoggingEnv, VecEnv, PytorchEnv, DictEnv

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}
REWARD_SPACES_DICT.update({
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
})
SUBTASKS_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.Subtask)
}
SUBTASK_SAMPLERS_DICT = {
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, multi_subtask.SubtaskSampler)
}


def create_flexible_obs_space(flags, teacher_flags: Optional) -> obs_spaces.BaseObsSpace:
    if teacher_flags is not None and teacher_flags.obs_space != flags.obs_space:
        # Train a student using a different observation space than the teacher
        return obs_spaces.MultiObs({
            "teacher_": teacher_flags.obs_space(**teacher_flags.obs_space_kwargs),
            "student_": flags.obs_space(**flags.obs_space_kwargs)
        })
    else:
        return flags.obs_space(**flags.obs_space_kwargs)


def _fixed_size_configuration(map_size: int) -> dict:
    """
    Base engine configuration pinned to a single map size.

    The engine treats width/height of -1 (the default) as "derive from the seed",
    which is why LuxEnv never fixes the board and the league sees a mix of
    {12, 16, 24, 32}. Setting both to a concrete value makes every episode that
    size, which is what a single-map-size experiment needs.
    """
    from kaggle_environments import make

    configuration = dict(make("lux_ai_2021").configuration)
    configuration["width"] = configuration["height"] = int(map_size)
    # LuxEnv only silences the engine on the branch where it builds the config
    # itself, so do it here too or every episode spams the log.
    configuration["loglevel"] = 0
    return configuration


def create_env(flags, device: torch.device, teacher_flags: Optional = None, seed: Optional[int] = None) -> DictEnv:
    if seed is None:
        seed = flags.seed
    map_size = getattr(flags, "map_size", None)
    base_configuration = _fixed_size_configuration(map_size) if map_size else None
    envs = []
    for i in range(flags.n_actor_envs):
        env = LuxEnv(
            act_space=flags.act_space(),
            obs_space=create_flexible_obs_space(flags, teacher_flags),
            # A private copy per env: LuxEnv mutates configuration["seed"] on every
            # reset, so a shared dict would make all n_actor_envs envs step the same
            # seed counter and play identical games.
            configuration=dict(base_configuration) if base_configuration else None,
            seed=seed
        )
        reward_space = create_reward_space(flags)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = PadFixedShapeEnv(env)
        env = LoggingEnv(env, reward_space)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env


def create_reward_space(flags) -> reward_spaces.BaseRewardSpace:
    if flags.reward_space is multi_subtask.MultiSubtask:
        assert "subtasks" in flags.reward_space_kwargs and "subtask_sampler" in flags.reward_space_kwargs
        subtasks = [SUBTASKS_DICT[s] for s in flags.reward_space_kwargs["subtasks"]]
        subtask_sampler = SUBTASK_SAMPLERS_DICT[flags.reward_space_kwargs["subtask_sampler"]]
        return flags.reward_space(subtasks, subtask_sampler)

    return flags.reward_space(**flags.reward_space_kwargs)
