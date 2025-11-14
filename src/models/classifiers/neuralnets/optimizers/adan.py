import tensorflow as tf
from typing import Any, Dict, List, Optional, Union


@tf.keras.saving.register_keras_serializable()
class Adan(tf.keras.optimizers.Optimizer):
    """ Keras‑native Adan optimizer (TensorFlow ≥ 2.11).

        Parameters
        ----------
        :param learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
                        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
                        that takes no arguments and returns the actual value to use. The
                        learning rate. Defaults to `0.001`.
        :param beta_1: A float value or a constant float tensor, or a callable
                that takes no arguments and returns the actual value to use. The
                exponential decay rate for the 1st moment estimates.
        :param beta_2: A float value or a constant float tensor, or a callable
                that takes no arguments and returns the actual value to use. The
                exponential decay rate for the 2nd moment estimates.
        :param beta_3: A float value or a constant float tensor, or a callable
                that takes no arguments and returns the actual value to use. The
                exponential decay rate for the 2nd alternative moment estimates.
        :param epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
                (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
        :param weight_decay: Float. If set, decoupled weight decay is applied (similar to AdamW).
        :param clipnorm: Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
        :param clipvalue: Float. If set, the gradient of each weight is clipped to be no higher than this value.
        :param global_clipnorm: Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.
        :param use_ema:	Boolean, defaults to False. If True, exponential moving average (EMA) is applied to the optimizer weights.
        :param ema_momentum:	Float, defaults to 0.99. Only used if use_ema=True. This is the momentum to use when computing the EMA of the model's weights.
        :param global_clipnorm: Float. If set, applies global gradient clipping.
        name: The name of the optimizer.
    """

    def __init__(
            self,
            learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
            beta_1: float = 0.98,
            beta_2: float = 0.92,
            beta_3: float = 0.99,
            epsilon: float = 1e-7,
            weight_decay: Optional[float] = None,
            clipnorm: Optional[float] = None,
            clipvalue: Optional[float] = None,
            global_clipnorm: Optional[float] = None,
            use_ema: bool = False,
            ema_momentum: float = 0.99,
            name='Adan',
            **kwargs
    ):
        # Validate arguments first.
        if not (isinstance(learning_rate, float) or isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule)):
            raise TypeError(f'learning_rate should either be float or WarmupLinearDecay, got {type(learning_rate)}')
        if epsilon < 0.0:
            raise ValueError("`epsilon` must be non-negative")
        if weight_decay == 0.0:
            weight_decay = None

        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum
        )

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = tf.cast(beta_1, tf.float32)
        self.beta_2 = tf.cast(beta_2, tf.float32)
        self.beta_3 = tf.cast(beta_3, tf.float32)
        self.epsilon = tf.cast(epsilon, tf.float32)

        self._exp_avg = None
        self._exp_avg_sq = None
        self._exp_avg_diff = None
        self._neg_pre_grad = None

    def build(self, var_list: List[tf.Tensor]):
        """ Create optimizer slot variables for `var_list`. """

        super().build(var_list)

        if getattr(self, "_built", False):
            return

        self._built = True

        self._exp_avg = []
        self._exp_avg_sq = []
        self._exp_avg_diff = []
        self._neg_pre_grad = []
        for v in var_list:
            self._exp_avg.append(self.add_variable_from_reference(model_variable=v, variable_name='exp_avg'))
            self._exp_avg_sq.append(self.add_variable_from_reference(model_variable=v, variable_name='exp_avg_sq'))
            self._exp_avg_diff.append(self.add_variable_from_reference(model_variable=v, variable_name='exp_avg_diff'))
            self._neg_pre_grad.append(self.add_variable_from_reference(model_variable=v, variable_name='neg_pre_grad'))

    def update_step(self, gradient: tf.Tensor, variable: tf.Tensor):
        """ Apply one update step for `variable` given its `gradient`. """

        if gradient is None:
            return

        # Fetch momentum, velocities from index.
        step = tf.cast(self.iterations + 1, tf.float32)
        idx = self._index_dict[self._var_key(variable=variable)]
        exp_avg = self._exp_avg[idx]
        exp_avg_sq = self._exp_avg_sq[idx]
        exp_avg_diff = self._exp_avg_diff[idx]
        neg_pre_grad = self._neg_pre_grad[idx]
        beta1 = self.beta_1
        beta2 = self.beta_2
        beta3 = self.beta_3
        lr = self.learning_rate

        # Calculate beta corrections.
        bias_correction1 = 1.0 - tf.pow(beta1, step)
        bias_correction2 = 1.0 - tf.pow(beta2, step)
        bias_correction3 = tf.sqrt(1.0 - tf.pow(beta3, step))

        # Sparse or Dense gradients branch.
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values           # shape [N, d]
            grad_indices = gradient.indices         # shape [N]

            def reset():
                return neg_pre_grad.assign(tf.zeros_like(gradient))

            def accumulate():
                return neg_pre_grad.scatter_add(tf.IndexedSlices(grad_values, grad_indices))

            # Reset or accumulate gradient differences.
            neg_grad_or_diff = tf.cond(tf.equal(self.iterations, 0), reset, accumulate)
            prev_diff_at_idx = tf.gather(neg_grad_or_diff, grad_indices)

            # ---- 1st moment (m_t) ---------------------------------------------------
            exp_avg.assign(exp_avg * beta1)
            exp_avg.scatter_add(tf.IndexedSlices((1.0 - beta1) * grad_values, grad_indices))

            # ---- diff  (d_t) without shape mixing  ---------------------------------
            exp_avg_diff.assign(exp_avg_diff * beta2)
            exp_avg_diff.scatter_add(tf.IndexedSlices((1.0 - beta2) * grad_values, grad_indices))
            exp_avg_diff.scatter_add(tf.IndexedSlices((1.0 - beta2) * prev_diff_at_idx, grad_indices))

            # ---- 2nd moment (v_t) on the slice -------------------------------------
            grad_diff = beta2 * prev_diff_at_idx + grad_values          # shape [N, d]
            exp_avg_sq.assign(exp_avg_sq * beta3)
            exp_avg_sq.scatter_add(tf.IndexedSlices((1.0 - beta3) * tf.square(grad_diff), grad_indices))

            # ---- parameter update only on touched rows -----------------------------
            m_t = tf.gather(exp_avg, grad_indices)
            v_t = tf.gather(exp_avg_sq, grad_indices)
            d_t = tf.gather(exp_avg_diff, grad_indices)

            denom = tf.sqrt(v_t) / bias_correction3 + self.epsilon
            step = lr / bias_correction1
            step_d = lr * beta2 / bias_correction2

            update_val = step * m_t / denom + step_d * d_t / denom
            variable.scatter_sub(tf.IndexedSlices(update_val, grad_indices))

            # ---- store −g_t for next iteration -------------------------------------
            neg_pre_grad.scatter_update(tf.IndexedSlices(-grad_values, grad_indices))
        else:
            # Update negative pre grad for diff
            def reset():
                return neg_pre_grad.assign(tf.zeros_like(gradient))

            def accumulate():
                return neg_pre_grad.assign_add(gradient)

            # Reset or accumulate gradient differences.
            neg_grad_or_diff = tf.cond(tf.equal(self.iterations, 0), reset, accumulate)

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg_t = exp_avg.assign(beta1 * exp_avg + (1.0 - beta1) * gradient)

            # diff_t = beta2 * diff_{t-1} + (1 - beta2) * (g_t + prev_diff)
            exp_avg_diff_t = exp_avg_diff.assign(beta2 * exp_avg_diff + (1.0 - beta2) * neg_grad_or_diff)

            # prepare for second moment: update neg_grad_or_diff again
            neg_grad_or_diff = neg_grad_or_diff*beta2 + gradient
            exp_avg_sq_t = exp_avg_sq.assign(beta3 * exp_avg_sq + (1.0 - beta3) * tf.square(neg_grad_or_diff))

            denom = tf.sqrt(exp_avg_sq_t) / bias_correction3 + self.epsilon
            step_size = lr / bias_correction1
            step_size_diff = lr * beta2 / bias_correction2

            # Parameter update
            update_val = (step_size * exp_avg_t / denom) + (step_size_diff * exp_avg_diff_t / denom)
            variable.assign_sub(update_val)

            # update neg_pre_grad to -g_t for next iteration
            neg_pre_grad.assign(-gradient)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                'learning_rate': self._serialize_hyperparameter(self._learning_rate),
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'beta_3': self.beta_3,
                'epsilon': self.epsilon,
            }
        )
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
