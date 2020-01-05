.. _callbacks:

Callbacks
=========

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to get a view on internal states and statistics of the RL model during training.
It allows to do monitoring, auto saving, model manipulation, progress bars, ...


A functional approach
---------------------

A callback function takes the `locals()` variables and the `globals()` variables from the model, then returns a boolean value for whether or not the training should continue.

Thanks to the access to the models variables, in particular `_locals["self"]`, we are able to even change the parameters of the model without halting the training, or changing the model's code.


.. code-block:: python

	def simple_callback(_locals, _globals):
		"""
		Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2).
		This callback will save the model and stop the training after the first call.

		:param _locals: (dict)
		:param _globals: (dict)
		"""
		print("callback called")
		# Save the model
		_locals["self"].save("saved_model")
		# If you want to continue training, the callback must return True.
		# return True # returns True, training continues.
		print("stop training")
		return False # returns False, training stops.


Object oriented approach
------------------------

.. code-block:: python

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

		def on_training_start(self, locals_: dict, globals_: dict) -> None:
			pass

		def on_step(self, locals_: dict, globals_: dict) -> bool:
			""" This method will be called by the model.
			This is the equivalent to the callback function.

			:param locals_: (dict)
			:param globals_: (dict)
			:return: (bool)
			"""
			return True

		def on_training_end(self, locals_: dict, globals_: dict) -> None:
			pass


Callback Collection
-------------------


.. automodule:: stable_baselines.common.callbacks
  :members:
