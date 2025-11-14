import tensorflow as tf
from typing import Any, Dict, List, Union, Optional


@tf.keras.saving.register_keras_serializable()
class AdaBelief(tf.keras.optimizers.Optimizer):
    """ Keras‑native **AdaBelief** optimizer with optional rectification. Implements J.Zhuang et al. (2020).
    
        Parameters
        ----------
        :param learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
                        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
                        that takes no arguments and returns the actual value to use. The
                        learning rate. Defaults to `0.001`.
        :param beta_1: A float value or a constant float tensor, or a callable
                that takes no arguments and returns the actual value to use. The
                exponential decay rate for the 1st moment estimates.
                Defaults to `0.9`.
        :param beta_2: A float value or a constant float tensor, or a callable
                that takes no arguments and returns the actual value to use. The
                exponential decay rate for the 2nd moment estimates.
                Defaults to `0.999`.
        :param epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
                (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
        :param weight_decay : Float. If set, decoupled weight decay is applied (similar to AdamW).
        :param amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Defaults to False.
        :param sma_threshold: A float value. The threshold for simple mean average.
        :param rectify: Whether to apply learning rate rectification as from RAdam.
        :param clipnorm: Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
        :param clipvalue: Float. If set, the gradient of each weight is clipped to be no higher than this value.
        :param global_clipnorm: Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.
        :param use_ema:	Boolean, defaults to False. If True, exponential moving average (EMA) is applied to the optimizer weights.
        :param ema_momentum:	Float, defaults to 0.99. Only used if use_ema=True. This is the momentum to use when computing the EMA of the model's weights.
        :param name: The name of the optimizer.
    """

    def __init__(
            self,
            learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-7,
            weight_decay: Optional[float] = None,
            amsgrad: bool = False,
            sma_threshold: float = 5.0,
            rectify: bool = True,
            clipnorm: Optional[float] = None,
            clipvalue: Optional[float] = None,
            global_clipnorm: Optional[float] = None,
            use_ema: bool = False,
            ema_momentum: float = 0.99,
            name: str = 'AdaBelief',
            **kwargs
    ):
        # Validate arguments first.
        if not (isinstance(learning_rate, float) or isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule)):
            raise TypeError(f'learning_rate should either be float or WarmupLinearDecay, got {type(learning_rate)}')
        if not 0 < beta_1 < 1.0:
            raise ValueError(f'beta1 parameter is expected to be in (0, 1.0), got {beta_1}.')
        if not 0 < beta_2 < 1.0:
            raise ValueError(f'beta1 parameter is expected to be in (0, 2.0), got {beta_2}.')
        if sma_threshold < 0:
            raise ValueError(f'SMA threshold is expected to be positive, got {sma_threshold}')
        if epsilon < 0.0:
            raise ValueError("`epsilon` must be non-negative")
        if weight_decay == 0.0:
            weight_decay = None

        super().__init__(
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            name=name
        )

        # Storing hyperparameters.
        self._learning_rate = self._build_learning_rate(learning_rate=learning_rate)
        self.beta_1 = tf.cast(beta_1, tf.float32)
        self.beta_2 = tf.cast(beta_2, tf.float32)
        self.epsilon = tf.cast(epsilon, tf.float32)
        self.weight_decay = tf.cast(weight_decay, tf.float32) if isinstance(weight_decay, float) else weight_decay
        self.amsgrad = amsgrad
        self.sma_threshold = tf.cast(sma_threshold, tf.float32)
        self.rectify = rectify

        # Declaring placeholders.
        self._momentums = None
        self._variances = None
        self._variances_hats = None

    def build(self, var_list: List[tf.Tensor]):
        """ Create optimizer‑specific variables for **each** model weight. """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True

        self._momentums = []        # Referenced as m in the original paper.
        self._variances = []        # Referenced as v in the original paper.
        self._variances_hats = []   # Referenced as v_hat in the original paper.
        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(model_variable=var, variable_name='m'))
            self._variances.append(self.add_variable_from_reference(model_variable=var, variable_name='v'))
            if self.amsgrad:
                self._variances_hats.append(self.add_variable_from_reference(model_variable=var, variable_name='vhat'))

    def update_step(self, gradient: tf.Tensor, variable: tf.Tensor):
        """ Performs one optimization step for each gradient, variable pair. """

        if gradient is None:
            return

        step = tf.cast(self.iterations + 1, tf.float32)
        beta1 = self.beta_1
        beta2 = self.beta_2
        eps = self.epsilon

        # Fetch momentum, velocities from index.
        idx = self._index_dict[self._var_key(variable=variable)]
        m = self._momentums[idx]
        v = self._variances[idx]
        b1_pow = tf.pow(beta1, step)
        b2_pow = tf.pow(beta2, step)

        # Sparse or dense branch.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse update
            grads = gradient.values
            indices = gradient.indices

            # Gather slot values
            m_slice = tf.gather(m, indices)
            v_slice = tf.gather(v, indices)

            # Update momentum and variance only on the sparse indices
            m_t_slice = beta1 * m_slice + (1 - beta1) * grads
            m_corr = m_t_slice / (1 - b1_pow)
            v_t_slice = beta2 * v_slice + (1 - beta2) * tf.square(grads - m_slice) + eps

            # Scatter updated slices back into full slots
            indices_expanded = tf.expand_dims(indices, axis=1)
            m.assign(tf.tensor_scatter_nd_update(m, indices_expanded, m_t_slice))
            v.assign(tf.tensor_scatter_nd_update(v, indices_expanded, v_t_slice))

            # Apply AMS Variant.
            if self.amsgrad:
                vhat = self._variances_hats[idx]
                vhat_slice = tf.gather(vhat, indices)
                vhat_updated = tf.maximum(vhat_slice, v_t_slice)
                vhat.assign(tf.tensor_scatter_nd_update(vhat, indices_expanded, vhat_updated))
                v_corr_slice = tf.sqrt(vhat_updated / (1.0 - b2_pow))
            else:
                v_corr_slice = tf.sqrt(v_t_slice / (1.0 - b2_pow))

            # Rectification (optional, as in RAdam)
            if self.rectify:
                sma_inf = 2.0 / (1.0 - beta2) - 1.0
                sma_t = sma_inf - 2.0 * step * b2_pow / (1.0 - b2_pow)
                r_t_num = (sma_t - 4.0) * (sma_t - 2.0) * sma_inf
                r_t_den = (sma_inf - 4.0) * (sma_inf - 2.0) * sma_t
                r_t = tf.sqrt(r_t_num / r_t_den)
                update_slice = tf.where(
                    sma_t >= self.sma_threshold,
                    r_t * m_corr / (v_corr_slice + eps),
                    m_corr
                )
            else:
                update_slice = m_corr/(v_corr_slice + eps)

            variable.assign(tf.tensor_scatter_nd_sub(variable, indices_expanded, self._get_learning_rate() * update_slice))
        else:
            # Compute moment estimates.
            m_t = m.assign(beta1 * m + (1.0 - beta1) * gradient)
            m_corr = m_t / (1 - b1_pow)
            v_t = v.assign(beta2 * v + (1.0 - beta2) * tf.square(gradient - m_t) + eps)

            # Apply AMS Variant.
            if self.amsgrad:
                vhat = self._variances_hats[idx]
                vhat.assign(tf.maximum(vhat, v_t))
                v_corr = tf.sqrt(vhat / (1.0 - b2_pow))
            else:
                v_corr = tf.sqrt(v_t / (1.0 - b2_pow))

            # Rectification (optional, as in RAdam)
            if self.rectify:
                sma_inf = 2.0 / (1.0 - beta2) - 1.0
                sma_t = sma_inf - 2.0 * step * b2_pow / (1.0 - b2_pow)
                r_t_numerator = (sma_t - 4.0) * (sma_t - 2.0) * sma_inf
                r_t_denominator = (sma_inf - 4.0) * (sma_inf - 2.0) * sma_t
                r_t = tf.sqrt(r_t_numerator / r_t_denominator)
                update = tf.where(
                    sma_t >= self.sma_threshold,
                    r_t * m_corr / (v_corr + eps),
                    m_corr
                )
            else:
                update = m_corr / (v_corr + eps)

            variable.assign_sub(self._get_learning_rate() * update)

    def _get_learning_rate(self) -> tf.Tensor:
        lr = self.learning_rate

        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return lr(self.iterations)

        return lr

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            'sma_threshold': self.sma_threshold,
            'rectify': self.rectify
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
