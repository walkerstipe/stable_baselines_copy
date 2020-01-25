.. _callbacks:

Callbacks
=========

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to access internal state of the RL model during training.
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...


A functional approach
---------------------

A callback function takes the `locals()` variables and the `globals()` variables from the model, then returns a boolean value for whether or not the training should continue.

Thanks to the access to the models variables, in particular `_locals["self"]`, we are able to even change the parameters of the model without halting the training, or changing the model's code.


.. code-block:: python

    from typing import Dict, Any

    from stable_baselines import PPO2


    def simple_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> bool:
        """
        Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2).
        This callback will save the model and stop the training after the first call.

        :param _locals: (Dict[str, Any])
        :param _globals: (Dict[str, Any])
        :return: (bool) If your callback returns False, training is aborted early.
        """
        print("callback called")
        # Save the model
        _locals["self"].save("saved_model")
        # If you want to continue training, the callback must return True.
        # return True # returns True, training continues.
        print("stop training")
        return False # returns False, training stops.

    model = PPO2('MlpPolicy', 'CartPole-v1')
    model.learn(2000, callback=simple_callback)


Object oriented approach
------------------------

This is the recommended approach.

.. code-block:: python

    from stable_baselines.common.callbacks import BaseCallback


    class CustomCallback(BaseCallback):
        """
        Base class for callback.

        :param verbose: (int)
        """
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            # Those variables will be accessible in the callback
            # (they are defined in the base class)
            # self.model = None
            # self.training_env = None
            # self.n_calls = 0
            # self.num_timesteps = 0
            # self.locals = None
            # self.globals = None

        def _on_training_start(self) -> None:
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model.
            This is the equivalent to the callback function.
            `locals()` and `globals()` are directly accessible as attributes.

            :return: (bool) Return True for continuing training, False for stopping.
            """
            return True

        def _on_training_end(self) -> None:
            pass


Callback Collection
-------------------

Stable Baselines provides you with a set of common callbacks.

CheckpointCallback
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import CheckpointCallback

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(2000, callback=checkpoint_callback)


EvalCallback
^^^^^^^^^^^^

For proper evaluation, using a separate test environment.

.. code-block:: python

    import gym

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import EvalCallback

    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=500)

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(5000, callback=eval_callback)


CallbackList
^^^^^^^^^^^^

For chaining callbacks.

.. code-block:: python

    import gym

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=500)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = SAC('MlpPolicy', 'Pendulum-v0')
    # Equivalent to:
    # model.learn(5000, callback=[checkpoint_callback, eval_callback])
    model.learn(5000, callback=callback)



.. automodule:: stable_baselines.common.callbacks
  :members:
