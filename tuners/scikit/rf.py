import pandas as pd
from models.scikit.rf import RandomForest
from tuners.tuner import Tuner


class RandomForestTuner(Tuner):
    def __init__(
        self,
        n_trials: int,
        metric,
        matches_df: pd.DataFrame,
        num_eval_samples: int,
        random_seed: int = 0,
    ):
        super().__init__(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            one_hot=False,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed,
        )

    def _create_model(self, trial) -> RandomForest:
        model = RandomForest(
            input_shape=self.x_train.shape[1:], random_seed=self.random_seed
        )

        n_estimators = trial.suggest_int("n_estimators", 50, 400)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        max_depth = trial.suggest_categorical(
            "max_depth", [None, 5, 10, 15, 20, 40, 50, 75, 90, 100]
        )
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 6])
        min_samples_split = trial.suggest_categorical(
            "min_samples_split", [2, 4, 5, 10]
        )
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        )
        is_calibrated = trial.suggest_categorical("is_calibrated", [False, True])

        model.build_model(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            class_weight=class_weight,
            is_calibrated=is_calibrated,
        )
        return model
