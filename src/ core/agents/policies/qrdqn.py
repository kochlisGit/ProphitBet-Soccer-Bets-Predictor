from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from sklearn.base import TransformerMixin
from src.core.agents.agent import Agent
from src.core.agents.environments.environment import BettingEnvironment
from src.core.preprocessing.utils.normalization import NormalizerType

if TYPE_CHECKING:
    from sb3_contrib.qrdqn import QRDQN


class QRDQNAgent(Agent):
    """ Base class for Stable-Baseline-based Quantile Regression Deep Q Networks (QR-DQN) policy models.

        Attributes
        ----------
        :param max_matches: (int) maximum number of weekly matches (rows) per fixture.
        :param normalizer: (str | NormalizerType | TransformMixIn) input data normalizer placeholder. If
                           NormalizerType provided, then the normalizer is built and stored once fit method is called.
        :param filters: (int) number of filters in the base model extractor.
        :param features_dim: (int) output dim (number of units) of the base model extractor.
        :param hidden_layers: (int) number of hidden layers for the policy networks.
        :param hidden_units: (int) number of neurons per hidden layer.
        :param activation: (str) activation function for the policy/value hidden layers.
        :param batch_size: (int) batch size of the network.
        :param learning_rate: (float) learning rate of the gradient descent algorithm.
        :param train_steps: (int) number of training steps for the algorithm.
        :param eval_freq: (int) evaluate the agent after eval_freq train timesteps.
        :param gamma: (float) discount factor (bounded between 0.0 and 1.0). Typically set to 0.99.
        :param initial_collect_steps: (int) number of steps to collect before learning starts.
        :param replay_buffer_size: (int) size of the replay buffer.
        :param target_update_freq: (int) update target network very target_update_freq steps.
        :param eps_fraction: (float) fraction of total training over which the linear annealing happens.
        :param eps_fraction: (float) random-action probability at the beginning.
        :param eps_initial: (float) random-action probability after annealing finishes.
    """

    def __init__(
            self,
            max_matches: int,
            normalizer: Optional[Union[str, NormalizerType, TransformerMixin]] = None,
            filters: int = 16,
            features_dim: int = 256,
            hidden_layers: int = 1,
            hidden_units: int = 256,
            activation: str = 'gelu',
            batch_size: int = 256,
            learning_rate: float = 5e-3,
            train_steps: int = 100000,
            eval_freq: int = 1000,
            gamma: float = 0.997,
            initial_collect_steps: int = 5000,
            replay_buffer_size: int = 50000,
            target_update_freq: int = 5000,
            eps_fraction: float = 0.005,
            eps_initial: float = 1.0,
            n_quantiles: int = 100
    ):
        from sb3_contrib.qrdqn import QRDQN

        super().__init__(
            max_matches=max_matches,
            algorithm=QRDQN,
            normalizer=normalizer,
            filters=filters,
            features_dim=features_dim,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_steps=train_steps,
            eval_freq=eval_freq,
            gamma=gamma,
            initial_collect_steps=initial_collect_steps,
            replay_buffer_size=replay_buffer_size,
            target_update_freq=target_update_freq,
            eps_fraction=eps_fraction,
            eps_initial=eps_initial,
            n_quantiles=n_quantiles
        )

    def build(self, env: BettingEnvironment, seed: int = 0) -> 'QRDQN':
        from src.core.agents.policies.base import BaseModel

        hidden_layers = [self._policy_params['hidden_units']]*self._policy_params['hidden_layers']
        return self._policy_params['algorithm'](
            policy='MultiInputPolicy',
            env=env,
            learning_rate=self._policy_params['learning_rate'],
            batch_size=self._policy_params['batch_size'],
            gamma=self._policy_params['gamma'],
            learning_starts=self._policy_params['initial_collect_steps'],
            buffer_size=self._policy_params['replay_buffer_size'],
            target_update_interval=self._policy_params['target_update_freq'],
            exploration_fraction=self._policy_params['eps_fraction'],
            exploration_initial_eps=self._policy_params['eps_initial'],
            n_steps=1,
            seed=seed,
            policy_kwargs={
                'features_extractor_class': BaseModel,
                'features_extractor_kwargs': {
                    'filters': self._policy_params['filters'],
                    'features_dim': self._policy_params['features_dim']
                },
                'net_arch': hidden_layers,
                'activation_fn': self._policy_params['activation'],
                'n_quantiles': self._policy_params['n_quantiles']
            }
        )

    @staticmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        return {
            'normalizer': [norm.value for norm in NormalizerType],
            'filters': {'low': 8, 'high': 64, 'step': 8},
            'features_dim': {'low': 64, 'high': 256, 'step': 64},
            'hidden_layers': {'low': 1, 'high': 2, 'step': 1},
            'hidden_units': {'low': 64, 'high': 256, 'step': 64},
            'activation': ['tanh', 'relu', 'gelu'],
            'batch_size': [32, 64, 128, 256],
            'learning_rate': {'low': 0.0005, 'high': 0.02, 'step': 0.0005},
            'gamma': {'low': 0.7, 'high': 0.99, 'step': 0.01},
            'initial_collect_steps': {'low': 5000, 'high': 15000, 'step': 2500},
            'target_update_freq': {'low': 2500, 'high': 15000, 'step': 2500},
            'eps_fraction': {'low': 0.005, 'high': 0.1, 'step': 0.005},
            'eps_initial': {'low': 0.7, 'high': 1.0, 'step': 0.05},
            'n_quantiles': {'low': 25, 'high': 200, 'step': 25},
        }
