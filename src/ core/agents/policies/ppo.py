import optuna
import pandas as pd
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from sklearn.base import TransformerMixin
from src.core.agents.agent import Agent
from src.core.agents.environments.environment import BettingEnvironment
from src.core.preprocessing.utils.normalization import NormalizerType

if TYPE_CHECKING:
    from stable_baselines3.ppo import PPO


class PPOAgent(Agent):
    """ Base class for Stable-Baseline-based Proximal Policy Optimization (PPO) policy models.

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
        :param n_steps: (int) number of environment steps to run before updating.
        :param epochs: (int) number of training epochs during update.
        :param gae_lambda: (float) generalized Advantage Estimator (GAE) factor. Typically set to 0.95.
        :param epsilon_clip: (float) policy clipping factor (bounded from 0.0 to 1.0).
        :param normalize_advantage: (bool) whether to normalize the advantage.
        :param entropy_regularization: (float): entropy regularization coefficient.
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
            n_steps: int = 1024,
            epochs: int = 15,
            gae_lambda: float = 0.95,
            epsilon_clip: float = 0.2,
            normalize_advantage: bool = True,
            entropy_regularization: float = 0.001
    ):
        from stable_baselines3.ppo import PPO

        super().__init__(
            max_matches=max_matches,
            algorithm=PPO,
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
            n_steps=n_steps,
            epochs=epochs,
            gae_lambda=gae_lambda,
            epsilon_clip=epsilon_clip,
            normalize_advantage=normalize_advantage,
            entropy_regularization=entropy_regularization
        )

    def build(self, env: BettingEnvironment, seed: int = 0) -> 'PPO':
        from src.core.agents.policies.base import BaseModel

        hidden_layers = [self._policy_params['hidden_units']]*self._policy_params['hidden_layers']
        return self._policy_params['algorithm'](
            policy='MultiInputPolicy',
            env=env,
            learning_rate=self._policy_params['learning_rate'],
            n_steps=self._policy_params['n_steps'],
            batch_size=self._policy_params['batch_size'],
            n_epochs=self._policy_params['epochs'],
            gamma=self._policy_params['gamma'],
            gae_lambda=self._policy_params['gae_lambda'],
            clip_range=self._policy_params['epsilon_clip'],
            normalize_advantage=self._policy_params['normalize_advantage'],
            ent_coef=self._policy_params['entropy_regularization'],
            seed=seed,
            policy_kwargs={
                'features_extractor_class': BaseModel,
                'features_extractor_kwargs': {
                    'filters': self._policy_params['filters'],
                    'features_dim': self._policy_params['features_dim']
                },
                'net_arch': {
                    'pi': hidden_layers,
                    'vf': hidden_layers
                },
                'activation_fn': self._policy_params['activation'],
                'ortho_init': True
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
            'epochs': {'low': 5, 'high': 30, 'step': 5},
            'gae_lambda': {'low': 0.9, 'high': 0.99, 'step': 0.01},
            'epsilon_clip': {'low': 0.1, 'high': 0.35, 'step': 0.05},
            'normalize_advantage': [False, True],
            'entropy_regularization': {'low': 0.0, 'high': 0.03, 'step': 0.001}
        }

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            seed: Optional[int] = 0,
            verbose: int = 1,
            trial: Optional[optuna.Trial] = None
    ) -> pd.DataFrame:
        report = super().train(train_df=train_df, eval_df=eval_df, seed=seed, verbose=verbose, trial=trial)
        self._policy.policy.action_dist.distribution = None
        return report
