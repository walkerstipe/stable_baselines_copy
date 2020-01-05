import os
from abc import ABC, abstractmethod
from typing import Union, List

import gym
import numpy as np

from stable_baselines.common.base_class import BaseRLModel # pytype: disable=pyi-error
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.evaluation import evaluate_policy


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: (int)
    """
    def __init__(self, verbose=0):
        super(BaseCallback, self).__init__()
        self.model = None
        self.training_env = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.verbose = verbose

    def init_callback(self, model: BaseRLModel) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        pass

    @abstractmethod
    def on_step(self, locals_: dict, globals_: dict) -> bool:
        """
        TODO: Should we modify current implementation?
        i.e. call after each env step instead after each rollout (current implementation)?

        :param locals_: (dict)
        :param globals_: (dict)
        :return: (bool)
        """
        return True

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        """
        This method will be called by the model. This is the equivalent to the callback function.
        :param locals_: (dict)
        :param globals_: (dict)
        :return: (bool)
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self.on_step(locals_, globals_)

    def on_training_end(self, locals_: dict, globals_: dict) -> None:
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def init_callback(self, model):
        super(CallbackList, self).init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(model)

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        for callback in self.callbacks:
            callback.on_training_start(locals_, globals_)

    def on_step(self, locals_, globals_):
        continue_training = True
        for callback in self.callbacks:
            # Update variables
            callback.num_timesteps = self.num_timesteps
            callback.n_calls = self.n_calls
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step(locals_, globals_) and continue_training
        return continue_training

    def on_training_end(self, locals_: dict, globals_: dict) -> None:
        for callback in self.callbacks:
            callback.on_training_end(locals_, globals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def init_callback(self, model: BaseRLModel) -> None:
        super(CheckpointCallback, self).init_callback(model)
        # Create folder if needed
        os.makedirs(self.save_path, exist_ok=True)

    def on_step(self, locals_: dict, globals_: dict) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True


class LambdaCallback(BaseCallback):
    """
    :param on_training_start: (callable)
    :param on_step: (callable)
    :param on_training_end: (callable)
    :param verbose: (int)
    """
    def __init__(self, on_training_start=None, on_step=None, on_training_end=None, verbose=0):
        super(LambdaCallback, self).__init__(verbose)
        if on_training_start is not None:
            self.on_training_start = on_training_start
        if on_step is not None:
            self._on_step = on_step
        else:
            self._on_step = lambda _locals, _globals: True
        if on_training_end is not None:
            self.on_training_end = on_training_end

    def on_step(self, locals_: dict, globals_: dict) -> bool:
        return self._on_step(locals_, globals_)


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param deterministic: (bool)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 n_eval_episodes=5, best_model_save_path=None,
                 eval_freq=10000, deterministic=True, verbose=1):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.deterministic = deterministic
        # TODO: check the env (num_envs == 1 and type(training_env) == type(eval_env))
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

    def on_step(self, locals_: dict, globals_: dict) -> bool:
        """
        :param locals_: (dict)
        :param globals_: (dict)
        :return: (bool)
        """

        if self.n_calls % self.eval_freq == 0:
            # TODO: sync training and eval env if there is VecNormalize
            episode_rewards, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                                 deterministic=self.deterministic, return_episode_rewards=True)


            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                # TODO: Save the agent if needed?
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                self.best_mean_reward = mean_reward

        return True
