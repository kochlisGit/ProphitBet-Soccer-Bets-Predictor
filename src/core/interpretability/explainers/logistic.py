import pandas as pd
import shap
from shap.maskers import Independent
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer


class LogisticRegressionExplainer(ClassifierExplainer):
    """ Explainer class for logistic regression models.

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

        explainer = shap.LinearExplainer(model=clf, masker=Independent(df, max_samples=df.shape[0]))
        return explainer(df)

    def get_coefficients(self) -> pd.DataFrame:
        """ Returns the coefficients of the linear model. """

        coefficients = self.model.get_coefficients()

        if len(self.target_names) < 3:
            return pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
        else:
            return pd.DataFrame(data=coefficients, columns=self.feature_names, index=self.target_names)
