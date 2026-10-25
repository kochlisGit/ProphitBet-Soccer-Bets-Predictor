import dtreeviz
import pandas as pd
import shap
from typing import Optional
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer
from src.core.models.classifiers.tree import DecisionTreeClassifier
from src.core.preprocessing.dataset import construct_targets


class DecisionTreeExplainer(ClassifierExplainer):
    """ Decision Tree explainer class. Supports only tree-based models.

        Attributes
        ----------
        :param df: (pd.DataFrame) league's dataframe.
        :param model_id: (str): model unique id.
        :param database: (ModelDatabase) active league's model database instance.
    """

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        super().__init__(df=df, model_id=model_id, database=database)

    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        clf = self.model.classifier

        if self.model_config.classifier_config['calibrate_probabilities']:
            clf = clf.calibrated_classifiers_[0].estimator

        if self.model_config.classifier_config['model_cls'] is DecisionTreeClassifier:
            explainer = shap.TreeExplainer(model=clf, data=df, feature_names=df.columns.tolist())
        else:
            explainer = shap.TreeExplainer(model=clf, feature_perturbation='tree_path_dependent', feature_names=df.columns.tolist())

        return explainer(df)


    def get_feature_importances(self) -> pd.DataFrame:
        """ Returns the coefficients of the tree-based model. """

        importances = self.model.get_importances()
        return pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})

    def get_tree_depth(self) -> Optional[int]:
        """Return the deepest fitted estimator used by the classifier."""

        clf = self.model.classifier
        if self.model_config.classifier_config.get('calibrate_probabilities', False):
            estimators = [
                calibrated.estimator
                for calibrated in clf.calibrated_classifiers_
            ]
        else:
            estimators = [clf]

        depths = [
            estimator.get_depth()
            for estimator in estimators
            if hasattr(estimator, 'get_depth')
        ]
        return max(depths) if depths else None

    def get_tree_model(self, tree_index: Optional[int] = None) -> Optional[dtreeviz.DTreeVizAPI]:
        """Return a dtreeviz wrapper for the fitted decision tree."""

        clf = self.model.classifier

        if self.model_config.classifier_config.get('calibrate_probabilities', False):
            clf = clf.calibrated_classifiers_[0].estimator

        try:
            x_train, _ = self._dataset_preprocessor.construct_inputs(df=self.train_df, normalizer=self.normalizer)
            y_train = construct_targets(df=self.train_df, target_type=self.target_type, one_hot=False)
            tree_model = dtreeviz.model(
                model=clf,
                X_train=x_train,
                y_train=y_train,
                tree_index=tree_index,
                feature_names=self.feature_names,
                target_name=self.target_type.value,
                class_names=self.target_names
            )
        except Exception as _:
            tree_model = None
        return tree_model
