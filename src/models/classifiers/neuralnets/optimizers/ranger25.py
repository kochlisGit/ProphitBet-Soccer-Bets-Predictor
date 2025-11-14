import tensorflow as tf
from typing import Any, Dict, List, Union, Optional


@tf.keras.saving.register_keras_serializable()
class Ranger25(tf.keras.optimizers.Optimizer):
    """ Keras‑native **Ranger25** optimizer as described in the reference paper. The components include:

        Components
        -----------------
        * Αdan
        * Cautious
        * StableAdamW or Adam-atan2
        * OrthoGrad
        * Adaptive gradient clipping
    """

    def __init__(
            self,
            learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
            beta_1: float = 0.98,
            beta_2: float = 0.92,
            beta_3: float = 0.99,
            epsilon: float = 1e-7,
            weight_decay: Optional[float] = 1e-4,
            adaptive_gradient_clipping: bool = True,
            adaptive_gradient_clipping_value: float = 1e-2,
            adaptive_gradient_clipping_eps: float = 1e-3,
            cautious: bool = True,
            stable_adamw: bool = True,
            adam_atan2: bool = True,
            orthograd: bool = True,
            orthograd_eps: float = 1e-16,
            name='ranger25',
            **kwargs
    ):
        # Validate arguments first.
        if not (isinstance(learning_rate, float) or isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule)):
            raise TypeError(f'learning_rate should either be float or WarmupLinearDecay, got {type(learning_rate)}')
        if not 0.0 < beta_1 < 1.0:
            raise ValueError(f'beta1 parameter is expected to be in (0, 1.0), got {beta_1}.')
        if not 0.0 < beta_2 < 1.0:
            raise ValueError(f'beta2 parameter is expected to be in (0, 2.0), got {beta_2}.')
        if not 0.0 < beta_3 < 1.0:
            raise ValueError(f'beta2 parameter is expected to be in (0, 2.0), got {beta_3}.')
        if epsilon < 0.0:
            raise ValueError('`epsilon` must be non-negative.')
        if adaptive_gradient_clipping_eps < 0.0:
            raise ValueError('`epsilon` must be non-negative.')

        super().__init__(name=name, weight_decay=weight_decay)

        # Storing hyperparameters.
        self._learning_rate = self._build_learning_rate(learning_rate=learning_rate)
        self.beta_1 = tf.cast(beta_1, tf.float32)
        self.beta_2 = tf.cast(beta_2, tf.float32)
        self.beta_3 = tf.cast(beta_3, tf.float32)
        self.epsilon = tf.cast(epsilon, tf.float32)
        self.agc = adaptive_gradient_clipping
        self.agc_value = tf.cast(adaptive_gradient_clipping_value, tf.float32)
        self.agc_eps = tf.cast(adaptive_gradient_clipping_eps, tf.float32)
        self.cautious = cautious
        self.stable_adamw = stable_adamw
        self.adam_atan2 = adam_atan2
        self.orthograd = orthograd
        self.orthograd_eps = orthograd_eps

        # Declaring placeholders.
        self._exp_avg = None
        self._exp_avg_sq = None
        self._exp_avg_diff = None
        self._neg_pre_grad = None

    def build(self, var_list: List[tf.Tensor]):
        """ Create optimizer‑specific variables for **each** model weight. """

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
        """ Performs one optimization step for each gradient, variable pair. """

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

        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)

        # Apply gradient orthogonalization.
        if self.orthograd:
            gradient = self._apply_orthogonal_gradients(gradient=gradient, variable=variable)

        # Apply adaptive gradient clipping.
        if self.agc:
            gradient = self._agc(gradient=gradient, variable=variable)

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

        # Apply stable-AdamW step_size clipping.
        if self.stable_adamw:
            step_size /= tf.clip_by_value(
                tf.sqrt(tf.reduce_mean(tf.pow(gradient, 2) / tf.maximum(exp_avg_sq, self.epsilon))),
                clip_value_min=1.0,
                clip_value_max=tf.float32.max
            )

        # Parameter update
        update_val = (step_size * exp_avg_t / denom) + (step_size_diff * exp_avg_diff_t / denom)

        # Apply cautious mechanism.
        if self.cautious:
            mask = tf.cast(tf.math.greater(exp_avg * gradient, 0.0), gradient.dtype)
            numel = tf.cast(tf.size(mask), gradient.dtype)
            factor = numel / (tf.reduce_sum(mask) + 1)
            mask = mask * factor
            update_val += update_val * mask

        variable.assign_sub(update_val)

        # Apply adam-atan2 mechanism.
        if self.adam_atan2:
            variable.assign_sub(tf.atan2(update_val, denom) * step_size)

        # update neg_pre_grad to -g_t for next iteration
        neg_pre_grad.assign(-gradient)

    def _l2_unit_norm(self, x: tf.Tensor) -> tf.Tensor:
        """ Applies l2-norm to gradients. """

        keepdims = True
        axis = None

        dims = x.shape.rank

        if dims <= 1:
            keepdims = False
        elif dims in (2, 3):
            axis = 1
        elif dims == 4:
            axis = (1, 2, 3)
        else:
            axis = tf.range(1, dims)

        return tf.norm(x, ord=2, axis=axis, keepdims=keepdims)

    def _agc(self, gradient: tf.Tensor, variable: tf.Tensor) -> tf.Tensor:
        """ Applies Adaptive Gradient Clipping to gradients. """

        max_norm = tf.maximum(self._l2_unit_norm(x=variable), self.agc_eps) * self.agc_value
        grad_norm = tf.maximum(self._l2_unit_norm(x=gradient), self.epsilon)
        clipped_grad = gradient * (max_norm / grad_norm)
        return tf.where(
            grad_norm > max_norm,
            clipped_grad,
            gradient
        )

    def _apply_orthogonal_gradients(self, gradient: tf.Tensor, variable: tf.Tensor) -> Union[tf.Tensor, tf.IndexedSlices]:
        """ Orthogonalizes gradients. """

        if isinstance(gradient, tf.IndexedSlices):
            return gradient

        w = tf.reshape(variable, [-1])
        g = tf.reshape(gradient, [-1])

        proj = tf.tensordot(w, g, axes=1) / (tf.tensordot(w, w, axes=1) + self.orthograd_eps)
        g_ortho = tf.cast(g, tf.float32) - proj * w
        g_norm = tf.norm(g)
        g_ortho_norm = tf.norm(g_ortho)
        g_ortho_scaled = g_ortho * (g_norm / (g_ortho_norm + self.orthograd_eps))
        return tf.reshape(g_ortho_scaled, tf.shape(gradient))

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'beta_3': self.beta_3,
            'epsilon': self.epsilon,
            'adaptive_gradient_clipping': self.adaptive_gradient_clipping,
            'adaptive_gradient_clipping_value': self.adaptive_gradient_clipping_value,
            'adaptive_gradient_clipping_eps': self.adaptive_gradient_clipping_eps,
            'cautious': self.cautious,
            'stable_adamw': self.stable_adamw,
            'adam_atan2': self.adam_atan2,
            'orthograd': self.orthograd,
            'orthograd_eps': self.orthograd_eps
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
