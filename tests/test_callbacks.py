from copy import deepcopy

import pytest

from stable_baselines import SAC
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


@pytest.mark.parametrize("model_class", [SAC])
def test_callbacks(model_class):
    model = model_class('MlpPolicy', 'Pendulum-v0')
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    # For testing: use the same training env
    eval_env = model.get_env()
    eval_callback = EvalCallback(eval_env, eval_freq=100)
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(1000, callback=callback)
