import os
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any

import gym
import numpy as np

from stable_baselines.common.base_class import BaseRLModel # pytype: disable=pyi-error
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization
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

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        pass

    # Should we include that?
    # def on_rollout_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
    #     pass

    @abstractmethod
    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        TODO: Should we modify current implementation?
        i.e. call after each env step instead after each rollout (current implementation)?

        :param locals_: (Dict[str, Any])
        :param globals_: (Dict[str, Any])
        :return: (bool)
        """
        return True

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        """
        This method will be called by the model. This is the equivalent to the callback function.
        :param locals_: (Dict[str, Any])
        :param globals_: (Dict[str, Any])
        :return: (bool)
        """
        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps + 1

        return self.on_step(locals_, globals_)

    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        pass

    # Should we include that?
    # def on_rollout_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
    #     pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def init_callback(self, model):
        super(CallbackList, self).init_callback(model)
        for callback in self.callbacks:
            callback.init_callback(model)

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
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

    def on_training_end(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
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
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
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

    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        return self._on_step(locals_, globals_)


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a log file (.npz) where the evaluations
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 n_eval_episodes=5, eval_freq=10000, log_path=None,
                 best_model_save_path=None,
                 deterministic=True, verbose=1):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.deterministic = deterministic
        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []

    def init_callback(self, model):
        super(EvalCallback, self).init_callback(model)
        # Does not work when eval_env is a gym.Env and training_env is a VecEnv
        # assert type(self.training_env) is type(self.eval_env), ("training and eval env are not of the same type",
        #                                                         "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_step(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:

        if self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                                 deterministic=self.deterministic, return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps, results=self.evaluations_results)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward

        return True
