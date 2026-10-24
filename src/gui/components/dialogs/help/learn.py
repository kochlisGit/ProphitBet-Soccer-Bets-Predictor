from __future__ import annotations

import webbrowser
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import ListView


class LearnDialog(QDialog):
    """Display tutorial links and open the selected item in a web browser."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._tutorials = [
            ('Statistical Learning', 'https://www.ibm.com/think/topics/statistical-machine-learning'),
            ('Machine Learning', 'https://developers.google.com/machine-learning/intro-to-ml/what-is-ml'),
            ('Classification', 'https://www.ibm.com/think/topics/classification-machine-learning'),
            ('Probabilistic Metrics', 'https://www.dratings.com/log-loss-vs-brier-score/'),
            ('Mean Implied Probability', 'https://www.sportmonks.com/glossary/implied-probability-converting-odds-into-a-percentage/'),
            ('Exploratory Data Analysis', 'https://www.epa.gov/caddis/exploratory-data-analysis'),
            ('Distributions', 'https://www.geeksforgeeks.org/maths/probability-distribution/'),
            ('Imbalances', 'https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets'),
            ('Correlations', 'https://www.jmp.com/en/statistics-knowledge-portal/linear-models/what-is-correlation'),
            ('Feature Selection', 'https://www.ibm.com/think/topics/feature-selection'),
            ('Explained Variance', 'https://www.statology.org/explained-variance/'),
            ('Boruta', 'https://blog.trainindata.com/is-boruta-dead/'),
            ('Coefficient Analysis', 'https://franciscobustamante.com.br/portfolio/2025-logistic-regression-coefficients/'),
            ('Rules Extraction', 'https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=nuggets-decision-tree-model-rules'),
            ('Classification Models', 'https://www.mathworks.com/campaigns/offers/next/choosing-the-best-machine-learning-classification-model-and-avoiding-overfitting.html'),
            ('Logistic Regression', 'https://www.ibm.com/think/topics/logistic-regression'),
            ('Naive Bayes Classifier', 'https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/'),
            ('K-Nearest Neighbors', 'https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/'),
            ('Linear Discriminant Analysis', 'https://www.ibm.com/think/topics/linear-discriminant-analysis'),
            ('Quadratic Discriminant Analysis', 'https://pinktra.com/learn/artificial-intelligence/machine-learning-algorithms/classification-algorithms/quadratic-discriminant-analysis-qda#real-world-applications'),
            ('Decision Tree', 'https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/'),
            ('Random Forest', 'https://www.ibm.com/think/topics/random-forest'),
            ('XGBoost', 'https://www.geeksforgeeks.org/machine-learning/xgboost/'),
            ('LightGBM', 'https://www.microsoft.com/en-us/research/project/lightgbm/'),
            ('Support Vector Machines', 'https://www.mathworks.com/discovery/support-vector-machine.html'),
            ('Neural Networks', 'https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_deep_neural_networks.htm'),
            ('Variable Selection Networks', 'https://www.sciencedirect.com/science/article/abs/pii/S0925231299001460'),
            ('Normalization', 'https://developers.google.com/machine-learning/crash-course/numerical-data/normalization'),
            ('Regularization', 'https://www.ibm.com/think/topics/regularization'),
            ('Undersampling', 'https://www.kaggle.com/discussions/general/424515'),
            ('Oversampling', 'https://blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/'),
            ('SHAP Explanations', 'https://www.geeksforgeeks.org/machine-learning/shap-a-comprehensive-guide-to-shapley-additive-explanations/'),
            ('Reinforcement Learning', 'https://www.ibm.com/think/topics/reinforcement-learning'),
            ('Deep Reinforcement Learning', 'https://www.ibm.com/think/topics/deep-reinforcement-learning'),
            ('Value-Based Methods', 'https://medium.com/data-science/value-based-methods-in-deep-reinforcement-learning-d40ca1086e1'),
            ('Policy Gradient Methods', 'https://www.datacamp.com/tutorial/policy-gradient-theorem'),
            ('Deep Q-Networks (DQN)', 'https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae'),
            ('Rainbow DQN', 'https://towardsdatascience.com/rainbow-dqn-the-best-reinforcement-learning-has-to-offer-166cb8ed2f86/'),
            ('QR-DQN', 'https://www.emergentmind.com/topics/quantile-regression-deep-q-network-qr-dqn'),
            ('Proximal Policy Optimization (PPO)', 'https://www.emergentmind.com/topics/proximal-policy-optimisation-ppo'),
            ('Hyperparameter Tuning', 'https://www.ibm.com/think/topics/hyperparameter-tuning'),
            ('Tree-structured Parzen Estimator', 'https://www.emergentmind.com/topics/tree-structured-parzen-estimators-tpe'),
            ('Gaussian Process', 'https://www.emergentmind.com/topics/optuna-based-hyperparameter-optimization'),
            ('Holdout Procedure', 'https://www.comet.com/site/blog/understanding-hold-out-methods-for-training-machine-learning-models/'),
            ('Stratified Holdout', 'https://medium.com/@sunils0506/types-of-cross-validations-f1ad3a871f67'),
            ('K-Fold Cross Validation', 'https://www.geeksforgeeks.org/machine-learning/k-fold-cross-validation-in-machine-learning/'),
            ('Nested Cross Validation', 'https://richard.dallaway.com/nested-cross-validation')
        ]

        self.setWindowTitle('Learn')
        self.setWindowIcon(QIcon('storage/icons/menu/help-learn.png'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setMinimumSize(500, 360)
        self.resize(680, 520)

        self._build_ui()

    @property
    def tutorials(self) -> list[tuple[str, str]]:
        return list(self._tutorials)

    @property
    def list_view(self) -> ListView:
        return self._list_view

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel('Tutorials', self)
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 3)
        title.setFont(title_font)
        layout.addWidget(title)

        description = QLabel('Select a tutorial to open it in your web browser.', self)
        description.setWordWrap(True)
        layout.addWidget(description)

        tutorial_names = [name for name, _url in self._tutorials]
        self._list_view = ListView(items=tutorial_names, parent=self)
        self._list_view.item_pressed.connect(self._open_tutorial)
        layout.addWidget(self._list_view, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton('Close', self)
        close_button.setMinimumSize(105, 36)
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)

    def _open_tutorial(self, tutorial_name: str) -> None:
        url = next(
            (url for name, url in self._tutorials if name == tutorial_name),
            None,
        )
        if not url:
            QMessageBox.warning(self, 'Tutorial Unavailable', 'The selected tutorial has no URL.')
            return

        if not webbrowser.open(url, new=2):
            QMessageBox.warning(
                self,
                'Browser Unavailable',
                f'Could not open the tutorial URL:\n{url}',
            )
