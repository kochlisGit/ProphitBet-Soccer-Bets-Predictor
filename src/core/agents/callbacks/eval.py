import copy
import optuna
import pandas as pd
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class EvalCallback(BaseCallback):
    """ Evaluates the current policy on the complete training and evaluation
        environments every eval_freq training timesteps. The parameters corresponding to the highest evaluation reward
        are preserved in memory.

        Attributes
        ----------
        :param train_env: (BettingEnvironment) The training environment.
        :param eval_env: (BettingEnvironment) The evaluation environment.
        :param eval_freq: (int) Evaluation frequency (number of train timesteps between evaluations).
        :param verbose: (int) verbosity. Defaults to 0.
    """

    def __init__(
            self,
            train_env: Monitor,
            eval_env: Optional[Monitor],
            eval_freq: int,
            trial: Optional[optuna.Trial] = None,
            verbose: int = 0
    ):
        super().__init__(verbose=verbose)

        self._train_env = train_env
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._trial = trial

        self._train_episodes = len(train_env.unwrapped.group_ids)
        self._eval_episodes = len(eval_env.unwrapped.group_ids) if self._eval_env is not None else 0

        self._steps = []
        self._train_rewards = []
        self._eval_rewards = []

        self._best_reward = -float('inf')
        self._best_parameters = None

    def _on_step(self) -> bool:
        """ Evaluates the policy model after eval_freq training timesteps and stores best model parameters. """

        if self.num_timesteps % self._eval_freq != 0:
            return True

        # Evaluating on the complete training dataframe.
        train_reward, _ = evaluate_policy(
            model=self.model,
            env=self._train_env,
            n_eval_episodes=self._train_episodes,
            deterministic=True
        )

        self._steps.append(self.num_timesteps)
        self._train_rewards.append(train_reward)

        # By default, use training reward for model selection.
        selection_reward = train_reward
        eval_reward = None

        # Evaluating on the evaluation dataframe, if provided.
        if self._eval_env is not None:
            eval_reward, _ = evaluate_policy(
                model=self.model,
                env=self._eval_env,
                n_eval_episodes=self._eval_episodes,
                deterministic=True
            )

            self._eval_rewards.append(eval_reward)
            selection_reward = eval_reward

        # Preserving the best policy in memory.
        if selection_reward > self._best_reward:
            self._best_reward = selection_reward
            self._best_parameters = copy.deepcopy(self.model.get_parameters())

        # Printing evaluation results.
        if self.verbose:
            message = f'Step: {self.num_timesteps} | Train Reward: {train_reward:.3f}'

            if eval_reward:
                message += f' | Eval Reward: {eval_reward:.3f}'

            print(message)

        # Reporting intermediate eval score to optuna.
        if self._trial is not None:
            self._trial.report(value=selection_reward, step=self.num_timesteps)

            if self._trial.should_prune():
                raise optuna.TrialPruned()

        return True

    def restore_best_model(self):
        """ Restores the parameters having the highest evaluation reward. """

        if self._best_parameters is not None:
            self.model.set_parameters(
                self._best_parameters,
                exact_match=True
            )

    def get_results(self) -> pd.DataFrame:
        """ Returns training/evaluation reward history. """

        results = {'Train Step': self._steps, 'Avg. Train Rewards': self._train_rewards}

        if self._eval_env is not None:
            results['Avg. Eval Rewards'] = self._eval_rewards

        return pd.DataFrame(results).round(decimals=3)
