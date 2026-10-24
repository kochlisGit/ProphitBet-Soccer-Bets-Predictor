import numpy as np
import pandas as pd
from math import prod
from typing import Any, Dict, Optional, Tuple, Type, Union
from tqdm import tqdm
from src.core.agents.agent import Agent, AgentConfig
from src.core.agents.environments.actions import Action
from src.core.agents.tuner import Tuner, TuningAlgorithm
from src.core.models.dashboard import get_study_filepath, get_storage_url
from src.core.preprocessing.utils.selection import train_test_split

BETTING_PERFORMANCE_METRICS = ['Bet ROI', 'Bet Win Rate']


def compute_max_weekly_matches(df: pd.DataFrame) -> int:
    """ Computes the maximum number of weekly matches for the provided dataframe.

        Args
        ----
        :param df: (pd.DataFrame) provided matches dataframe.
    """

    group_sizes = df.groupby(['Season', 'Week']).size()
    max_matches = group_sizes.max()
    return max_matches


def create_agent(
        league_id: str,
        agent_id: str,
        df: pd.DataFrame,
        test_size: Union[float, int],
        agent_cls: Type[Agent],
        agent_params: Dict[str, Any],
        tunable_params: Dict[str, Any],
        trials: int = 50,
        tune_algorithm: TuningAlgorithm = TuningAlgorithm.TPE,
        seed: Optional[int] = 0,
        open_optuna_dashboard: bool = True
) -> Tuple[AgentConfig, Agent]:
    """ Constructs an agent as follows:

        Steps:
            1. Dataset is split into train-test.
            2. Fine-tuning is applied on the training set.
            3. Training is applied on train set.
            4. The policy is evaluated on the training and evaluation sets.

        Args
        ----
        :param league_id: (str) league id.
        :param agent_id: (str) agent's id.
        :param df: (Pandas DataFrame) dataset to be used for training & tuning.
        :param test_size: (Union[float, int]) number of samples to be used for validation.
        :param agent_cls: (Type[Agent]) agent (policy) class.
        :param agent_params: (Dict[str, Any]) agent parameters.
        :param tunable_params: (Dict[str, Any]) tunable agent parameters.
        :param trials: (int) number of trials (hyperaparameter search combinations). Defaults to 50.
        :param tune_algorithm: Tuning Algorithm instance. Defaults to TPE.
        :param seed: (int) random seed for the experiments. Defaults to 0.
        :param open_optuna_dashboard: (bool) whether to open optuna dashboard.

        :returns: Constructed agent config and the trained agent.
    """

    report = {'league_id': league_id, 'agent_id': agent_id}

    # Dropping missing rows.
    valid_df = df.dropna()

    # Computing max weekly matches.
    max_matches = compute_max_weekly_matches(df=valid_df)
    agent_params['max_matches'] = max_matches

    # Splitting dataset into train-test.
    train_df, eval_df = train_test_split(df=valid_df, test_size=test_size)

    # Fine-tuning.
    if tunable_params:
        tuner = Tuner(agent_cls=agent_cls, agent_params=agent_params, tunable_params=tunable_params)

        storage_url = get_storage_url(study_filepath=get_study_filepath(league_id=league_id, entity='agents'))
        report['tune_report'], _, best_params, report['tune_param_importances'] = tuner.tune(
            df=train_df,
            test_size=test_size,
            trials=trials,
            algorithm=tune_algorithm,
            storage_url=storage_url,
            study_name=agent_id,
            seed=seed,
            open_optuna_dashboard=open_optuna_dashboard
        )

        agent_params.update(best_params)

    # Training-Evaluating agent.
    agent = agent_cls(**agent_params)
    report['train_report'] = agent.train(train_df=train_df, eval_df=eval_df, seed=seed)
    report['agent_config'] = agent.policy_params
    report['num_train_samples'] = train_df.shape[0]
    report['num_eval_samples'] = eval_df.shape[0]
    agent_config = AgentConfig(**report)
    return agent_config, agent


def compute_betting_metrics(df: pd.DataFrame, actions: np.ndarray, units: float = 1.0) -> Tuple[float, float]:
    """ Compute the mean ROI (Return of Investment) across weekly fixtures, and the average win_rate.

        Action ``0`` skips a match. Actions ``1``, ``2`` and ``3`` are single
        bets on home, draw and away, respectively. Actions ``4``, ``5`` and
        ``6`` add the corresponding selection to the week's accumulator. A
        losing single costs ``units`` and a losing accumulator costs ``units``
        once, regardless of its leg count.

        Args
        ----
        df:
            Match data containing ``Season``, ``Week``, ``Result`` and the
            ``1``, ``X`` and ``2`` odds columns.
        actions:
            One action per dataframe row.
        units:
            Stake used for each single bet and weekly accumulator.

        Returns
        -------
        average_roi:
            Mean ROI across weekly fixtures.

        win_rate:
            Total correct bets / total staked bets.
            SKIP actions are ignored.
            Each accumulator counts as one bet.
    """

    if df.empty:
        return 0.0, 0.0

    if len(actions) != df.shape[0]:
        raise ValueError(f'Expected one action per match, got {len(actions)} actions for {df.shape[0]} matches.')

    reward_df = df[['Season', 'Week', 'Result', '1', 'X', '2', '1X', '12', 'X2']].copy()
    reward_df['Action'] = actions

    fixture_rois = []
    fixture_win_rates = []
    for _, fixture_df in tqdm(iterable=reward_df.groupby(['Season', 'Week'], sort=True), desc='Fixture'):
        payout = 0.0
        staked = 0.0
        total_bets = 0
        winning_bets = 0

        accumulator_bets = []
        for _, match in fixture_df.iterrows():
            action = match['Action']

            if action == Action.SKIP.value:
                continue
            else:
                if action == Action.SOLO_1.value:
                    success = match['Result'] == 'H'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['1'] - 1.0) if success else -units
                elif action == Action.SOLO_X.value:
                    success = match['Result'] == 'D'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['X'] - 1.0) if success else -units
                elif action == Action.SOLO_2.value:
                    success = match['Result'] == 'A'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['2'] - 1.0) if success else -units
                elif action == Action.SOLO_1X.value:
                    success = match['Result'] == 'H' or match['Result'] == 'D'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['1X'] - 1.0) if success else -units
                elif action == Action.SOLO_12.value:
                    success = match['Result'] == 'H' or match['Result'] == 'A'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['12'] - 1.0) if success else -units
                elif action == Action.SOLO_X2.value:
                    success = match['Result'] == 'D' or match['Result'] == 'A'
                    total_bets += 1
                    staked += units
                    winning_bets += int(success)
                    payout += units * (match['X2'] - 1.0) if success else -units
                elif action == Action.ACCUMULATOR_1.value:
                    accumulator_bets.append((match['1'], match['Result'] == 'H'))
                elif action == Action.ACCUMULATOR_X.value:
                    accumulator_bets.append((match['X'], match['Result'] == 'D'))
                elif action == Action.ACCUMULATOR_2.value:
                    accumulator_bets.append((match['2'], match['Result'] == 'A'))
                elif action == Action.ACCUMULATOR_1X.value:
                    accumulator_bets.append((match['1X'], match['Result'] == 'H' or match['Result'] == 'D'))
                elif action == Action.ACCUMULATOR_12.value:
                    accumulator_bets.append((match['12'], match['Result'] == 'H' or match['Result'] == 'A'))
                elif action == Action.ACCUMULATOR_X2.value:
                    accumulator_bets.append((match['X2'], match['Result'] == 'D' or match['Result'] == 'A'))
                else:
                    raise NotImplementedError(f'Not implemented action: "{action}".')

        if accumulator_bets:
            total_bets += 1
            staked += units
            if all(success for _, success in accumulator_bets):
                winning_bets += 1
                odd_cummul = prod([odd for odd, _ in accumulator_bets])
                payout += units * (odd_cummul - 1)
            else:
                payout -= units

        fixture_roi = payout/staked if staked > 0 else 0.0
        fixture_rois.append(fixture_roi)

        win_rate = winning_bets/total_bets if total_bets > 0 else 0.0
        fixture_win_rates.append(win_rate)

    average_roi = round(float(np.mean(fixture_rois)), 3)
    average_win_rate = round(float(np.mean(fixture_win_rates)), 3)
    return average_roi, average_win_rate
