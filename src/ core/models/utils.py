import pandas as pd
from typing import Any, Dict, Optional, Tuple, Type, Union
from src.core.models.dashboard import get_study_filepath, get_storage_url
from src.core.models.model import Classifier, ModelConfig
from src.core.models.trainer import Trainer
from src.core.models.tuner import Tuner, OptimizationMetric, TuningAlgorithm
from src.core.preprocessing.utils.selection import train_test_split


def create_model(
        league_id: str,
        model_id: str,
        df: pd.DataFrame,
        test_size: Union[float, int],
        model_cls: Type,
        model_params: Dict[str, Any],
        tunable_params: Dict[str, Any],
        trials: int = 100,
        tune_algorithm: TuningAlgorithm = TuningAlgorithm.TPE,
        tune_metric: OptimizationMetric = OptimizationMetric.ACC,
        tune_nested_cv: bool = False,
        tune_fast_mode: bool = False,
        cross_validation: bool = False,
        sliding_cross_validation: bool = False,
        seed: Optional[int] = 0,
        open_optuna_dashboard: bool = True
) -> Tuple[ModelConfig, Classifier]:
    """ Constructs a model as follows:

        Steps:
            1. Dataset is split into train-test.
            2. Fine-tuning is applied on train set (either using nested cross-validation, cross-validation or holdout).
            3. Training is applied on train set.
            4. Model is evaluated in train set using cross validation and sliding cross validation,
               and the train-test metrics.

        Args
        ----
        :param league_id: (str) league id.
        :param model_id: (str) model id.
        :param df: (Pandas DataFrame) dataset to be used for training & tuning.
        :param test_size: (Union[float, int]) number of samples to be used for validation.
        :param model_cls: (Type[Classifier]) model class.
        :param model_params: (Dict[str, Any]) model parameters.
        :param tunable_params: (Dict[str, Any]) tunable model parameters.
        :param trials: (int) number of trials (hyperaparameter search combinations). Defaults to 100.
        :param tune_algorithm: Tuning Algorithm instance. Defaults to TPE.
        :param tune_metric: (OptimizationMetric) tuning metric. Defaults to ACC.
        :param tune_nested_cv: (bool) whether to use nested cross validation. Uses max 100 trials. Defaults to False.
        :param tune_fast_mode: (bool) whether to use fast mode for tuning, nested-cv. Defaults to False.
        :param cross_validation: (bool) whether to use cross validation to evaluate model. Defaults to False.
        :param sliding_cross_validation: (bool) whether to use sliding cross validation to evaluate model.
               Defaults to False.
        :param seed: (int) random seed for the experiments. Defaults to 0.
        :param open_optuna_dashboard: (bool) whether to open optuna dashboard.

        :returns: Constructed model config and trained classifier.
    """

    report = {'league_id': league_id, 'model_id': model_id}

    # Dropping missing rows.
    valid_df = df.dropna()

    # Splitting dataset into train-test.
    train_df, test_df = train_test_split(df=valid_df, test_size=test_size)

    # Fine-tuning.
    if tunable_params:
        tuner = Tuner(model_cls=model_cls, model_params=model_params, tunable_params=tunable_params, metric=tune_metric)

        if tune_nested_cv:
            report['nested_cv_tune_report'], _, report['tune_param_importances'] = tuner.nested_cv_tuning(
                df=train_df,
                trials=min(trials, 100),
                algorithm=tune_algorithm,
                fast_mode=tune_fast_mode,
                seed=seed
            )

        storage_url=get_storage_url(study_filepath=get_study_filepath(league_id=league_id, entity='models'))
        report['tune_report'], _, best_params, importances = tuner.tune(
            df=train_df,
            trials=trials,
            algorithm=tune_algorithm,
            return_importances=not tune_nested_cv,
            fast_mode=tune_fast_mode,
            storage_url=storage_url,
            study_name=model_id,
            seed=seed,
            open_optuna_dashboard=open_optuna_dashboard
        )

        if not 'tune_param_importances' in report:
            report['tune_param_importances'] = importances

        model_params.update(best_params)

    # Evaluating the model using cross-validation and sliding cross-validation techniques.
    trainer = Trainer()
    if cross_validation:
        report['cv_report'] = trainer.cross_validation(
            df=train_df,
            model_cls=model_cls,
            model_config=model_params,
            show_progress=True,
            seed=seed
        )
    if sliding_cross_validation:
        test_ratio = test_size if isinstance(test_size, float) else round(test_df.shape[0]/train_df.shape[0], 2)
        report['sliding_cv_report'] = trainer.sliding_cross_validation(
            df=train_df,
            model_cls=model_cls,
            model_config=model_params,
            test_ratio=test_ratio,
            show_progress=True,
            seed=seed
        )

    # Training-Evaluating model.
    model = model_cls(**model_params)
    report['train_report'] = model.fit(train_df=train_df, eval_df=test_df, seed=seed)
    report['classifier_config'] = model.model_config
    report['num_train_samples'] = train_df.shape[0]
    report['num_eval_samples'] = test_df.shape[0]
    model_config = ModelConfig(**report)
    return model_config, model
