import random
import gymnasium as gym
import numpy as np
from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import TransformerMixin
from src.core.agents.environments.actions import Action
from src.core.preprocessing.dataset import DatasetPreprocessor
from src.core.preprocessing.utils import NormalizerType, TargetType


class BettingEnvironment(gym.Env):
    """ Custom betting environment that follows gym interface. The environment is connected to Stable Baseline's
        framework. It requires an env_config attribute with the following keys:
            * *df*: (pd.DataFrame) league dataframe.
            * *normalizer*: (str | NormalizerType | TransformerMixin | None) input normalizer instance.
            * *shuffle*: whether to shuffle weekly fixtures. It is useful during training mode.
            * *seed*: environment's seed that enable reproducibility. Defaults to None.

        Observation Space
        ------------------
        At each timestep t, the agent receives 3 observations:
            1. Current (weekly) fixture data, including the model probabilities for Result: NxF,
               where N is number of matches and F is number of features.
            2. Current fixture step (match index) as embedding (N,).
            3. Constructed Betting strategy: (Nx13) as embedding.
        So it's a Dict[str, Box] space.

        **NOTE: The matrices are transposed to be compatible with the pytorch last-channel operations.**

        Action Space
        ------------
        At each timestep t decides for a match one of a following actions:
            0. SKIP
            1. SOLO-1
            2. SOLO-X
            3. SOLO-2
            4. SOLO-1X
            5. SOLO-12
            6. SOLO-X2
            7. MULTIPLE-1
            8. MULTIPLE-X
            9. MULTIPLE-2
            10. MULTIPLE-1X
            11. MULTIPLE-12
            12. MULTIPLE-X2
        So it's a discrete space of size (13,).

        Reward Function
        ---------------
        The payout (total profit from all bets) is computed as reward function. Each bet consumes U=1.0 budget.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()

        df = env_config['df']
        self._normalizer = env_config['normalizer']
        self._shuffle = env_config['shuffle']
        self._max_matches = env_config['max_weekly_matches']

        seed = env_config.get('seed', None)
        if seed is not None:
            random.seed(seed)

        # Validating dataframe.
        if df.isna().any().any():
            raise ValueError('Invalid dataframe: found missing values. Remove nan values before initializing env.')

        required_columns = ['Season', 'Week', '1', 'X', '2', '1X', '12', 'X2', TargetType.RESULT.value]
        if not set(required_columns).issubset(df.columns):
            raise KeyError(f'Expected columns: "{required_columns}", found only: {df.columns}')

        # Normalizing inputs.
        dataset_preprocessor = DatasetPreprocessor()
        x, y, self._normalizer = dataset_preprocessor.preprocess(
            df=df,
            target_type=TargetType.RESULT,
            normalizer=self._normalizer
        )
        odds = df[['1', 'X', '2', '1X', '12', 'X2']].to_numpy(dtype=np.float32)

        # Grouping fixtures by week.
        self._weekly_fixtures = []
        self._weekly_results = []
        self._weekly_odds = []
        self._weekly_fixture_sizes = []

        df_indices = list(df.groupby(by=['Season', 'Week'], sort=True).indices.values())
        for idx in df_indices:
            self._weekly_fixtures.append(x[idx].T)
            self._weekly_results.append(y[idx])
            self._weekly_odds.append(odds[idx])
            self._weekly_fixture_sizes.append(idx.shape[0])
        self._df_index = np.hstack(df_indices)

        if max(self._weekly_fixture_sizes) > self._max_matches:
            raise ValueError(
                f'The defined max_matches has exceeded: Expected fixture matches <= {self._max_matches}, got '
                f'{max(self._weekly_fixture_sizes)}'
            )

        # Defining step encodings and group ids.
        self._encoded_steps = {i: np.eye(self._max_matches, dtype=np.float32)[i] for i in range(self._max_matches)}

        # Declaring placeholders.
        self._current_fixture = None
        self._current_results = None
        self._current_odds = None
        self._current_step = None
        self._current_strategy = None
        self._group_ids = None
        self._current_fixture_size = None
        self._payout = 0.0
        self._staked = 0.0
        self._winning_bets = 0
        self._total_bets = 0.0
        self._accumulator_bets = []

        # Computing group ids.
        self._initialize_group_ids()

        # Environment Parameters.
        self._units = 1.0

        # Defining action space.
        self.action_space = gym.spaces.Discrete(n=len(Action))

        # Defining observation space.
        low = np.minimum(x.min(axis=0), 0.0)   # (F,)
        high = np.maximum(x.max(axis=0), 0.0)  # (F,)
        fixture_low = np.tile(low[:, None], (1, self._max_matches))
        fixture_high = np.tile(high[:, None], (1, self._max_matches))
        self.observation_space = gym.spaces.Dict(spaces={
            'Fixture': gym.spaces.Box(
                low=fixture_low,
                high=fixture_high,
                shape=(x.shape[1], self._max_matches),
                dtype=np.float32
            ),
            'Step': gym.spaces.Box(low=0.0, high=1.0, shape=(self._max_matches,), dtype=np.float32),
            'Strategy': gym.spaces.Box(low=0.0, high=1.0, shape=(len(Action), self._max_matches), dtype=np.float32),
        })

    @property
    def normalizer(self) -> Optional[Union[NormalizerType, TransformerMixin]]:
        return self._normalizer

    @property
    def df_index(self) -> np.ndarray:
        return self._df_index

    @property
    def group_ids(self) -> List[int]:
        return self._group_ids

    @property
    def max_matches(self) -> int:
        return self._max_matches

    def _initialize_group_ids(self):
        """ Initializes group ids, which will be used to fetch the next fixture during reset. """

        self._group_ids = list(range(len(self._weekly_fixtures)))

        if self._shuffle:
            random.shuffle(self._group_ids)

    def reset(self, seed: Optional[int] = None, options=None):
        """ Resets environment to its initial state. """

        super().reset(seed=seed, options=options)

        # Initializing group ids from scratch, if the entire dataset has been fed.
        if not self._group_ids:
            self._initialize_group_ids()

        # Fetching the next fixture data.
        i = self._group_ids.pop(0)

        self._current_fixture = self._weekly_fixtures[i].copy()
        self._current_results = self._weekly_results[i].copy()
        self._current_odds = self._weekly_odds[i].copy()
        self._current_fixture_size = self._weekly_fixture_sizes[i]

        # Re-arranging match order.
        if self._shuffle:
            perm = self.np_random.permutation(self._current_fixture_size)

            self._current_fixture = self._current_fixture[:, perm]
            self._current_results = self._current_results[perm]
            self._current_odds = self._current_odds[perm]

        # Padding fixture after shuffling the real matches.
        if self._current_fixture.shape[1] != self._max_matches:
            self._current_fixture = np.pad(
                self._current_fixture,
                ((0, 0), (0, self._max_matches - self._current_fixture.shape[1])),
                mode='constant',
                constant_values=0
            )

        self._current_step = 0
        self._current_strategy = np.zeros(shape=(len(Action), self._max_matches), dtype=np.float32)

        # Resetting betting metrics and placeholders.
        self._payout = 0.0
        self._staked = 0.0
        self._winning_bets = 0
        self._total_bets = 0.0
        self._accumulator_bets = []

        observation = {
            'Fixture': self._current_fixture,
            'Step': self._encoded_steps[self._current_step],
            'Strategy': self._current_strategy
        }
        info = {'Step': self._current_step, 'Results': self._current_results}
        return observation, info

    def step(self, action: int) -> tuple[Dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """ Applies the selected action (bet) to the environment,
            computes the rewards and transits into the next state.

            Args
            ----
            :param action: (int) selected action by the agent.
        """

        if action != Action.SKIP.value:
            outcome = self._current_results[self._current_step]
            odds = self._current_odds[self._current_step]

            if action == Action.SOLO_1.value:
                success = outcome == 0
                odd = odds[0]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.SOLO_X.value:
                success = outcome == 1
                odd = odds[1]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.SOLO_2.value:
                success = outcome == 2
                odd = odds[2]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.SOLO_1X.value:
                success = (outcome == 0 or outcome == 1)
                odd = odds[3]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.SOLO_12.value:
                success = (outcome == 0 or outcome == 2)
                odd = odds[4]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.SOLO_X2.value:
                success = (outcome == 1 or outcome == 2)
                odd = odds[5]

                self._total_bets += 1.0
                self._winning_bets += int(success)
                reward = self._units * (odd - 1.0) if success else -self._units
                self._payout += reward
                self._staked += self._units
            elif action == Action.ACCUMULATOR_1.value:
                reward = 0.0
                self._accumulator_bets.append((odds[0], outcome == 0))
            elif action == Action.ACCUMULATOR_X.value:
                reward = 0.0
                self._accumulator_bets.append((odds[1], outcome == 1))
            elif action == Action.ACCUMULATOR_2.value:
                reward = 0.0
                self._accumulator_bets.append((odds[2], outcome == 2))
            elif action == Action.ACCUMULATOR_1X.value:
                reward = 0.0
                self._accumulator_bets.append((odds[3], outcome == 0 or outcome == 1))
            elif action == Action.ACCUMULATOR_12.value:
                reward = 0.0
                self._accumulator_bets.append((odds[4], outcome == 0 or outcome == 2))
            elif action == Action.ACCUMULATOR_X2.value:
                reward = 0.0
                self._accumulator_bets.append((odds[5], outcome == 1 or outcome == 2))
            else:
                raise NotImplementedError(f'Not implemented action: "{action}".')
        else:
            reward = 0.0

        # Updating current strategy.
        self._current_strategy[:, self._current_step] = Action.to_one_hot_vector(action=action)

        # Incrementing step by 1.
        self._current_step += 1
        done = self._current_step == self._current_fixture_size

        if done:
            if self._accumulator_bets:
                self._total_bets += 1
                self._staked += self._units

                if all(success for _, success in self._accumulator_bets):
                    self._winning_bets += 1
                    odd_cummul = prod([odd for odd, _ in self._accumulator_bets])
                    accumulator_reward = self._units*(odd_cummul - 1)
                else:
                    accumulator_reward = -self._units

                self._payout += accumulator_reward
                reward += accumulator_reward

            encoded_step = np.zeros(shape=(self._max_matches,), dtype=np.float32)
            roi = self._payout/self._staked if self._staked > 0 else 0.0
            win_rate = self._winning_bets/self._total_bets if self._total_bets > 0 else 0.0
        else:
            encoded_step = self._encoded_steps[self._current_step]
            roi = win_rate = None

        # Computing next state.
        observation = {'Fixture': self._current_fixture, 'Step':  encoded_step, 'Strategy': self._current_strategy}
        truncated = False
        info = {
            'Step': self._current_step,
            'Metrics': {'Bet ROI': roi, 'Bet Win Rate': win_rate}
        }
        return observation, reward, done, truncated, info

    def render(self):
        super().render()

    def close(self):
        super().close()
