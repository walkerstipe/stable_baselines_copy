from typing import Union
from copy import deepcopy

import gym

# flake8: noqa F401
from stable_baselines.common.vec_env.base_vec_env import AlreadySteppingError, NotSteppingError, VecEnv, VecEnvWrapper, \
    CloudpickleWrapper
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines.common.vec_env.vec_check_nan import VecCheckNan


def unwrap_vec_normalize(env: Union[gym.Env, VecEnv]) -> Union[VecNormalize, None]:
    """
    :param env: (Union[gym.Env, VecEnv])
    :return: (VecNormalize)
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


# Define here to avoid circular import
def sync_envs_normalization(env: Union[gym.Env, VecEnv], eval_env: Union[gym.Env, VecEnv]):
    """
    Sync eval and train environments when using VecNormalize

    :param env: (Union[gym.Env, VecEnv]))
    :param eval_env: (Union[gym.Env, VecEnv]))
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            # No need to sync the reward scaling
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv
