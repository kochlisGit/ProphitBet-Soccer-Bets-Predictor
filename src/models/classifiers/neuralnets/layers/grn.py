import tensorflow as tf
from typing import Optional, Tuple


@tf.keras.saving.register_keras_serializable()
class GatedResidualNetwork(tf.keras.layers.Layer):
    """ Implementation of Gated Residual Network (GRN).
        y1 = dense(x) -> elu -> dropout -> dense
        y2 = sigmoid(x)
        residual = skip(x)
        scores = y1 * y2
        y = scores + residual -> layer norm


        It passes each input through a dense layer and computes a score for each input.

        Parameters
        --------------------------------------------

        :param hidden_units: Number of hidden units.
        :param output_units: Number of output units.
        :param dropout_rate: Whether to apply dropout regularization with the specified dropout rate.
    """

    def __init__(self, hidden_units: int, output_units: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)

        self._output_units = output_units

        self.linear_in = tf.keras.layers.Dense(units=hidden_units, activation='elu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.linear_out = tf.keras.layers.Dense(units=output_units, activation=None)
        self.gate = tf.keras.layers.Dense(units=output_units, activation='sigmoid')
        self.skip_projection = None
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape: Tuple):
        if input_shape[-1] != self._output_units:
            self.skip_projection = tf.keras.layers.Dense(units=self._output_units, activation=None, use_bias=False)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # --- Main transformation path. ---
        x = self.linear_in(inputs)
        x = self.dropout(x, training=training)
        x = self.linear_out(x)

        # --- Gating mechanism. ---
        gate = self.gate(inputs)
        x = x * gate

        # --- Add residual connection & normalize. ---
        residual = self.skip_projection(inputs) if self.skip_projection else inputs
        y = residual + x
        y = self.layer_norm(y)
        return y
