import numpy as np
import optuna
import pandas as pd
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union
from sklearn.base import TransformerMixin
from tqdm import tqdm
from src.core.agents.environments.environment import BettingEnvironment
from src.core.preprocessing.utils.normalization import NormalizerType

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm


class Agent(ABC):
    """ Base class wrapper for Stable-Baseline-based policy models.

        Properties
        -----------

        * All policies inside this class are treated as Stable-Baseline policies.
        * Each policy uses a BaseModel as feature extractor.

        Attributes
        ----------
        :param max_matches: (int) maximum number of weekly matches (rows) per fixture.
        :param algorithm (Type): policy's algorithm class.
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
        :param gamma: (float) discount factor (bounded between 0.0 and 1.0).
    """

    def __init__(
            self,
            max_matches: int,
            algorithm: Type,
            normalizer: Optional[Union[str, NormalizerType, TransformerMixin]] = None,
            filters: int = 16,
            features_dim: int = 128,
            hidden_layers: int = 1,
            hidden_units: int = 256,
            activation: Union[str, Type[torch.nn.Module]] = 'gelu',
            batch_size: int = 256,
            learning_rate: float = 5e-3,
            train_steps: int = 100000,
            eval_freq: int = 500,
            gamma: float = 0.99,
            **kwargs
    ):
        if activation == 'gelu' or activation is torch.nn.GELU:
            activation = torch.nn.GELU
        elif activation == 'relu' or activation is torch.nn.ReLU:
            activation = torch.nn.ReLU
        elif activation == 'tanh' or activation is torch.nn.Tanh:
            activation = torch.nn.Tanh
        else:
            raise ValueError(f'Not supported activated function: "{activation}"')

        self._policy_params = {
            'agent_cls': self.__class__,
            'env': None,
            'max_matches': max_matches,
            'algorithm': algorithm,
            'normalizer': normalizer,
            'filters': filters,
            'features_dim': features_dim,
            'hidden_layers': hidden_layers,
            'hidden_units': hidden_units,
            'activation': activation,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_steps': train_steps,
            'eval_freq': eval_freq,
            'gamma': gamma,
            **kwargs
        }
        self._policy = None
        self._decimals = 3

    @property
    def policy_params(self) -> Dict[str, Any]:
        """Return persisted parameters without the derived algorithm class."""

        config = self._policy_params.copy()
        config.pop('algorithm', None)
        return config

    @property
    def policy(self) -> Optional['BaseAlgorithm']:
        return self._policy

    @staticmethod
    def from_config(agent_config: Dict[str, Any]) -> 'Agent':
        """ Reconstructs an agent wrapper from a stored agent configuration. """

        config = agent_config.copy()
        agent_cls = config.pop('agent_cls', None)
        config.pop('env', None)
        config.pop('algorithm', None)
        return agent_cls(**config)

    @abstractmethod
    def build(self, env: BettingEnvironment, seed: int = 0) -> 'BaseAlgorithm':
        """ Builds a selected classifier. It should be a BaseEstimator instance.

            :param env: (BettingEnvironment) gymnasium-based environment.
            :param seed: (int) model's random seed, to enable result reproducibility.
        """

        pass

    @staticmethod
    @abstractmethod
    def get_suggested_hyperparameters() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """ Returns a suggested set of all hyperparameters (Dict[str, values]) for the classifier.  """

        pass

    def load(self, checkpoint_directory: str, env: BettingEnvironment):
        """ Loads the trained policy from its checkpoint. Override for custom loading methods.

            Args
            ----
            :param checkpoint_directory: (str) filepath of the checkpoint directory.
            :param env: (BettingEnvironment) environment used to restore the policy model.
        """

        checkpoint_fp = f'{checkpoint_directory}/policy.zip'
        self._policy = self._policy_params['algorithm'].load(checkpoint_fp, env=env)

    def save(self, checkpoint_directory: str):
        """ Saves the trained policy into the specified location. Override for custom saving methods.

            Args
            ----
            :param checkpoint_directory: (str) filepath of the checkpoint directory.
        """

        if self._policy is None:
            raise ValueError('Policy model has not been built/trained yet. Cannot save None policy.')

        checkpoint_fp = f'{checkpoint_directory}/policy.zip'
        self._policy.save(checkpoint_fp)

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            seed: Optional[int] = 0,
            verbose: int = 1,
            trial: Optional[optuna.Trial] = None
    ) -> pd.DataFrame:
        """ Trains the agent using the provided train dataframe and evaluates it in the evaluation dataframe.
            The training iterations are specified in the constructor of this class.

            :param train_df: (DataFrame) training dataframe.
            :param eval_df: (DataFrame) evaluation dataframe.
            :param seed: (int | None) random seed for reproducibility. Defaults to 0.
            :param verbose: (int) verbosity of the training. Defaults to 1, which prints the train-evaluation results.
                            Set 0 to silence.
            :param trial: (optuna.Trial) optuna's trial instance.
        """
        if not (verbose == 0 or verbose == 1):
            raise ValueError(f'verbose is expected to be 0 for Silence or 1 for Evaluation Logging, got {verbose}.')

        from stable_baselines3.common.monitor import Monitor
        from src.core.agents.callbacks.eval import EvalCallback

        # Building the training environment.
        train_env = BettingEnvironment(env_config={
            'df': train_df,
            'normalizer': self._policy_params['normalizer'],
            'max_weekly_matches': self._policy_params['max_matches'],
            'shuffle': True,
            'seed': seed,
        })
        self._policy_params['normalizer'] = train_env.normalizer
        self._policy_params['env'] = train_env

        # Building the training-evaluation environment and its monitor callback.
        train_eval_env = Monitor(
            env=BettingEnvironment(env_config={
                'df': train_df,
                'normalizer': train_env.normalizer,
                'max_weekly_matches': self._policy_params['max_matches'],
                'shuffle': False,
                'seed': seed
            })
        )

        eval_env = None

        if eval_df is not None:
            eval_env = Monitor(
                env=BettingEnvironment(env_config={
                    'df': eval_df,
                    'normalizer': train_env.normalizer,
                    'max_weekly_matches': self._policy_params['max_matches'],
                    'shuffle': False,
                    'seed': seed
                })
            )

        eval_callback = EvalCallback(
            train_env=train_eval_env,
            eval_env=eval_env,
            eval_freq=min(self._policy_params['eval_freq'], self._policy_params['train_steps']),
            trial=trial,
            verbose=verbose,
        )

        # Initializing policy model & starting training process.
        self._policy = self.build(env=train_env, seed=seed)
        self._policy.learn(
            total_timesteps=self._policy_params['train_steps'],
            callback=eval_callback,
            progress_bar=bool(verbose)
        )

        eval_callback.restore_best_model()
        return eval_callback.get_results()

    def evaluate(self, env: BettingEnvironment) -> Tuple[List[int], float, Dict[str, float]]:
        """ Evaluates the trained policy model on the given environment.

            Returns the generated actions, mean monetary reward per fixture,
            and mean betting metrics across fixtures.
        """

        actions = []
        rewards = []
        metrics = {}
        for _ in tqdm(iterable=range(len(env.group_ids)), desc='Fixture'):
            obs, _ = env.reset()
            done = truncated = False

            fixture_reward = 0.0
            info = {}
            while not (done or truncated):
                action, _ = self._policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action=action)

                actions.append(action)
                fixture_reward += reward

            rewards.append(fixture_reward)

            for metric_name, metric_value in info['Metrics'].items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(metric_value)

        avg_reward = round(float(np.mean(rewards)), 3)
        avg_metrics = {metric_name: round(float(np.mean(values)), 3) for metric_name, values in metrics.items()}
        return actions, avg_reward, avg_metrics

    def compute_actions(self, df: pd.DataFrame) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """ Given a fixture or multiple fixtures grouped by (season, week), it computes the optimal betting strategy.
            Returns the actions per weekly fixture along with the total average rewards.

            :param df: (pd.DataFrame) fixture dataframe.
        """

        env = BettingEnvironment(env_config={
            'df': df,
            'normalizer': self._policy_params['normalizer'],
            'max_weekly_matches': self._policy_params['max_matches'],
            'shuffle': False
        })
        actions, rewards, metrics = self.evaluate(env=env)

        order = np.argsort(env.df_index)
        actions = np.array(actions, dtype=np.int32)[order]
        return actions, rewards, metrics


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """ Agent's config class that stores all user parameters about a built policy.

        Attributes
        ----------
        :param league_id: (str) user-specified league identifier.
        :param agent_id: (str) user-specified model identifier for the selected league.
        :param agent_config: (Dict[str, Any]) configuration required to reconstruct the agent.
        :param num_train_samples: (int) number of samples used for training.
        :param num_eval_samples: (int) number of samples used for evaluation.
        :param train_report: pd.DataFrame model's training metrics.
        :param tune_report: (pd.DataFrame | None) whether hyperparameter optimization has been applied.
        :param tune_param_importances: (Dict[str, float] | None) parameter importances extracted during the tuning process.
    """

    league_id: str
    agent_id: str
    agent_config: Dict[str, Any]
    num_train_samples: int
    num_eval_samples: int
    train_report: pd.DataFrame
    tune_report: Optional[pd.DataFrame] = False
    tune_param_importances: Optional[Dict[str, float]] = None
