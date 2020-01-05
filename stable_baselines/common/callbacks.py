from abc import ABC, abstractmethod
from typing import Union, List

import gym
import numpy as np

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

    def init_callback(self, model: Union[gym.Env, VecEnv]) -> None:
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

    @abstractmethod
    def on_training_end(self, locals_: dict, globals_: dict) -> None:
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def init_callback(self, model):
        for callback in self.callbacks:
            callback.init_callback(model)

    def on_training_start(self, model):
        for callback in self.callbacks:
            callback.on_training_start(model)

    def on_step(self, locals_, globals):
        continue_training = True
        for callback in self.callbacks:
            continue_training = callback.on_step(locals_, globals_) and continue_training
        return continue_training

    def on_training_end(self, model):
        for callback in self.callbacks:
            callback.on_training_end(model)


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
