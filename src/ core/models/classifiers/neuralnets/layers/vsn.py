import tensorflow as tf
from typing import Optional, Tuple
from src.core.models.classifiers.neuralnets.layers.grn import GatedResidualNetwork


@tf.keras.saving.register_keras_serializable()
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """ Implementation of Variable Selection Network (VSN).
        It passes each input feature through GRN and assigns its GRN a score, based on the feature importance.


        Attributes:
        --------------------------------------------
        :param num_inputs: (int) Number of input featuers.
        :param hidden_units: (int) Number of projected hidden units.
        :param dropout_rate: (float) Whether to apply dropout regularization ε with the specified dropout rate 0 < ε < 1.0.
    """

    def __init__(self, input_size: int, hidden_units: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)

        self._input_size = input_size

        self.feature_grns = [
            GatedResidualNetwork(hidden_units=hidden_units, output_units=hidden_units, dropout_rate=dropout_rate, name=f'feat_grn_{i+1}')
            for i in range(input_size)
        ]
        self.weight_grns = [
            GatedResidualNetwork(hidden_units=hidden_units, output_units=1, dropout_rate=dropout_rate, name=f'weight_grn_{i+1}')
            for i in range(input_size)
        ]
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        transformed_vars = []
        weights = []

        # Iterate over variables along feature dimension and pass each variable through feat, weight GRNs.
        feat_splits = tf.split(value=inputs, num_or_size_splits=self._input_size, axis=-1)
        for i, (feat_grn, weight_grn, feat) in enumerate(zip(self.feature_grns, self.weight_grns, feat_splits)):
            transformed_feat = feat_grn(feat, training=training)                # (B, H)
            transformed_vars.append(transformed_feat)
            weight = weight_grn(feat, training=training)                        # (B, 1)
            weights.append(weight)

        # Stack transformed variables, weights.
        transformed_stack = tf.stack(values=transformed_vars, axis=1)           # (B, F, H)
        weight_stack = tf.concat(values=weights, axis=1)                        # (B, F)

        # Softmax to get attention / selection probabilities
        attn_weights = self.softmax(weight_stack)                               # (B, F)

        # Weighted sum over feature dimension.
        attn_weights_exp = tf.expand_dims(attn_weights, -1)                     # (B, F, 1)
        context = tf.reduce_sum(attn_weights_exp*transformed_stack, axis=1)     # (B, H)
        return context, attn_weights
