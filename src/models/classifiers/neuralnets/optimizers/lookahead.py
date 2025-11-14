import tensorflow as tf
from typing import Any, Dict, List, Union


@tf.keras.saving.register_keras_serializable()
class Lookahead(tf.keras.optimizers.Optimizer):
    """ KERAS-native Lookahead optimizer wrapper (TensorFlow ≥ 2.11). Keeps a copy of updates (slow updates), which
        it synchronizes after N update steps.

        Parameters
        ----------
        :param optimizer: tf.keras.optimizers.Optimizer | str | dict
            The *inner* optimizer that performs the fast updates.  You can pass an
            actual optimizer instance, a serialized config `dict`, or a string that
            `tf.keras.optimizers.get` understands (e.g. "SGD").
        :param sync_period: int, default 6
            Number of **inner** optimizer steps before the *slow* weights are
            synchronized with the *fast* weights.
        :param slow_step_size: float, default 0.5
            Interpolation factor for the update of the slow weights.
            After *k* inner steps the slow weights are moved towards the fast ones
            by `slow_step_size` :slow = slow + slow_step_size * (fast - slow)
        :param name: str, default "Lookahead"
            Python identifier for the wrapper.
    """

    def __init__(
            self,
            optimizer: Union[str, tf.keras.optimizers.Optimizer],
            sync_period: int = 6,
            slow_step_size: float = 0.5,
            name: str = 'Lookahead',
            **kwargs
    ):
        # Validate arguments first.
        if sync_period <= 1:
            raise ValueError('sync_period must be > 1.')
        if not 0.0 < slow_step_size <= 1.0:
            raise ValueError('slow_step_size must be in (0, 1].')

        # Deserialize inner optimizer if necessary.
        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(f'optimizer must be instance of tf.keras.optimizers.Optimizer, got "{type(optimizer)}".')

        super().__init__(name=name)

        # Storing hyperparameters.
        self._base_optimizer = optimizer
        self._learning_rate = tf.cast(1.0, tf.float32)
        self.sync_period = tf.cast(sync_period, tf.int64)
        self.slow_step_size = tf.cast(slow_step_size, tf.float32)

        # Declaring placeholders.
        self._slow_vars = None

    @property
    def variables(self) -> List[tf.Tensor]:
        """ Returns slow weights + fast weights"""
        return super().variables + self._base_optimizer.variables

    @property
    def learning_rate(self) -> float:
        return self._base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._base_optimizer.learning_rate = learning_rate

    @property
    def lr(self) -> float:
        return self.learning_rate

    @lr.setter
    def lr(self, lr: float):
        self.learning_rate = lr

    def build(self, var_list: List[tf.Tensor]):
        """ Creates optimizer‑specific variables for **each** model weight and creates slow variables, which is a copy of the original vars. """

        # Build state for the outer wrapper itself (index‑dict etc.).
        super().build(var_list=var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True

        # Build slow variables.
        self._slow_vars = [
            self.add_variable_from_reference(model_variable=v, variable_name='slow', initial_value=v)
            for v in var_list
        ]

    def update_step(self, gradient: tf.Tensor, variable: tf.Tensor):
        """ Performs an update step given the calculated gradients. """

        if gradient is None:
            return

        # Fetch slow variables from index.
        slow_var = self._slow_vars[self._index_dict[self._var_key(variable=variable)]]

        # Synchronize fast-slow weights.
        def _synchronize():
            """ Move slow weights towards fast, then copy back to fast. """

            new_slow = slow_var + self.slow_step_size * (variable - slow_var)
            slow_var.assign(new_slow)
            variable.assign(new_slow)

        tf.cond(tf.equal((self.iterations + 1) % self.sync_period, 0), _synchronize, lambda: None)

    def apply_gradients(
            self,
            grads_and_vars: List[tf.Tensor],
            name: str = None,
            skip_gradients_aggregation: bool = False,
            **kwargs,
    ):
        """ Updates optimizer's iteration counter and applies gradients. """

        if not hasattr(self, "_built"):
            with tf.name_scope(name or self.name):
                with tf.init_scope():
                    _, var_list = zip(*grads_and_vars)
                    self.build(var_list=var_list)

        # Apply base optimizer gradients.
        train_op = self._base_optimizer.apply_gradients(
            grads_and_vars=grads_and_vars,
            name=self._base_optimizer.name,
            skip_gradients_aggregation=skip_gradients_aggregation,
            **kwargs
        )
        super().apply_gradients(grads_and_vars=grads_and_vars, name=name, skip_gradients_aggregation=skip_gradients_aggregation, **kwargs)
        return train_op

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'optimizer': tf.keras.optimizers.serialize(self._base_optimizer),
            'sync_period': self.sync_period,
            'slow_step_size': self.slow_step_size,
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inner_cfg = config.pop('optimizer')
        inner_opt = tf.keras.optimizers.deserialize(inner_cfg)
        return cls(inner_opt, **config)
