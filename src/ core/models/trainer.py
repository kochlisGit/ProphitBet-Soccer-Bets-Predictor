import pandas as pd
from typing import Any, Dict, Optional, Tuple, Type, Union
from src.core.preprocessing.utils import construct_targets
from src.core.preprocessing.utils.selection import train_test_split, cross_validation, sliding_cross_validation
from src.core.models.model import Classifier


class Trainer:
    """ Trainer-Evaluator class for classification models. """

    def train(
            self,
            model: Classifier,
            df: pd.DataFrame,
            test_size: Union[int, float],
            seed: Optional[int] = 0
    ) -> Tuple[Classifier, pd.DataFrame]:
        """ Fits the model in the provided dataset. Returns the train-eval metrics.

            Args
            ----
            :param model: (Classifier) classifier to be trained.
            :param df: (pd.DataFrame) dataframe with league's data. It requires Nan rows to be removed.
            :param test_size: (int | float) size of the test set.
            :param seed: (int | None) model's random seed, to enable result reproducibility.
        """

        train_df, eval_df = train_test_split(df=df, test_size=test_size)
        metrics_df = model.fit(train_df=train_df, eval_df=eval_df, seed=seed)
        return model, metrics_df

    def cross_validation(
            self,
            df: pd.DataFrame,
            model_cls: Type,
            model_config: Dict[str, Any],
            k_folds: int = 5,
            show_progress: bool = False,
            seed: Optional[int] = 0
    ) -> pd.DataFrame:
        """ Evaluates the model using the Stratify k-fold cross validation method and returns the metrics per fold.

            Args
            ----
            :param df: (pd.DataFrame) training dataframe.
            :param model_cls: (Type[Classifier]) classifier class to be trained.
            :param model_config: (Dict[str, Any]) model parameters.
            :param k_folds: (int) number of folds to be used for cross validation.
            :param show_progress: (bool) whether to show the progress bar or not.
            :param seed: (int) random seed of the selection process.
        """

        targets = construct_targets(df=df, target_type=model_config['target_type'])

        metrics_per_fold = []
        for i, train_df, eval_df in cross_validation(
                df=df,
                targets=targets,
                k_folds=k_folds,
                show_progress=show_progress,
                seed=seed
        ):
            model = model_cls(**model_config)
            metrics_df = model.fit(train_df=train_df, eval_df=eval_df, seed=seed)
            metrics_df.insert(loc=0, column='Fold', value=i+1)
            metrics_per_fold.append(metrics_df)

        metrics_df = pd.concat(metrics_per_fold, axis=0, ignore_index=True)
        return metrics_df

    def sliding_cross_validation(
            self,
            df: pd.DataFrame,
            model_cls: Type,
            model_config: Dict[str, Any],
            test_ratio: float,
            k_folds: int = 5,
            show_progress: bool = False,
            seed: Optional[int] = 0
    ) -> pd.DataFrame:
        """ Evaluates the model using the Sliding k-fold cross validation method.

            Args
            ----
            :param df: (pd.DataFrame) training dataframe.
            :param model_cls: (Type[Classifier]) classifier class to be trained.
            :param model_config: (Dict[str, Any]) model parameters.
            :param test_ratio: portion of the dataset to be used as test set.
            :param k_folds: (int) number of folds to be used for cross validation.
            :param show_progress: (bool) whether to show the progress bar or not.
            :param seed: (int) random seed of the selection process.
        """

        metrics_per_fold = []
        for i, train_df, eval_df in sliding_cross_validation(
                df=df,
                test_ratio=test_ratio,
                k_folds=k_folds,
                show_progress=show_progress
        ):
            model = model_cls(**model_config)
            metrics_df = model.fit(train_df=train_df, eval_df=eval_df, seed=seed)
            metrics_df.insert(loc=0, column='Fold', value=i+1)
            metrics_per_fold.append(metrics_df)

        metrics_df = pd.concat(metrics_per_fold, axis=0, ignore_index=True)
        return metrics_df
