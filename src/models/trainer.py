import math
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import construct_targets
from src.models.model import ClassificationModel


class Trainer:
    """ Classification Model trainer class. """

    def train(
            self,
            model: ClassificationModel,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            check_nan: bool = True
    ) -> Tuple[ClassificationModel, pd.DataFrame]:
        """ Fits the model in the provided dataset. """

        # Clean Dataframe.
        if check_nan and (train_df.isna().any().any() or eval_df.isna().any().any()):
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        metrics_df = model.fit(train_df=train_df, eval_df=eval_df)
        return model, metrics_df

    def cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using the Stratify k-fold cross validation method. """

        # Clean Dataframe.
        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        targets = construct_targets(df=df, target_type=model.target_type)

        # Initialize stratified k-fold procedure.
        metrics_df_per_fold = []
        cv_generator = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0).split(df, targets)
        for i, (train_ids, eval_ids) in tqdm(iterable=enumerate(cv_generator), desc='Evaluating K-Fold', total=k_folds):
            train_df = df.iloc[train_ids]
            eval_df = df.iloc[eval_ids]
            model, metrics_df = self.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = i + 1
            metrics_df_per_fold.append(metrics_df)
        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df

    def sliding_cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            test_ratio: float,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using the Sliding k-fold cross validation method. """

        # Rectify the k-folds if case samples are too few.
        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        samples_per_fold = int(math.floor(df.shape[0]/k_folds))

        # Initialize sliding k-fold procedure.
        metrics_df_per_fold = []
        for i in tqdm(iterable=range(k_folds), desc='Evaluating Sliding K-Fold', total=k_folds):
            fold_df = df.iloc[-(i+1)*samples_per_fold:]
            train_df, eval_df = train_test_split(df=fold_df, test_size=test_ratio)
            model, metrics_df = self.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = i + 1
            metrics_df['Start Date'] = [train_df.iloc[-1]['Date'], eval_df.iloc[-1]['Date']]
            metrics_df['End Date'] = [train_df.iloc[0]['Date'], eval_df.iloc[0]['Date']]
            metrics_df['Samples'] = [train_df.shape[0], eval_df.shape[0]]
            metrics_df_per_fold.append(metrics_df)
        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df
