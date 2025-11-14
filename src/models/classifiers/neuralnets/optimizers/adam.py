import tensorflow as tf
from typing import Optional


@tf.keras.saving.register_keras_serializable()
class Adam(tf.keras.optimizers.Adam):
    """ TF-Keras Adam optimizer wrapper that ensures consistency with the entire framework. """

    def __init__(
            self,
            learning_rate: float = 0.001,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-7,
            amsgrad: bool = False,
            weight_decay: Optional[float] = None,
            clipnorm: Optional[float] = None,
            clipvalue: Optional[float] = None,
            global_clipnorm: Optional[float] = None,
            use_ema: bool = False,
            ema_momentum: float = 0.99,
            ema_overwrite_frequency: Optional[int] = None,
            jit_compile: bool = True,
            name: str = 'Adam',
            **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            name=name
        )
