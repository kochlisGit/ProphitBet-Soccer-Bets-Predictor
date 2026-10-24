from importlib import import_module
from typing import Any, Dict, Optional
from PySide6.QtWidgets import QGridLayout, QMessageBox, QWidget
from src.gui.components.workspaces.page import WorkspaceActionCard, WorkspacePage


class ModelsWorkspace(WorkspacePage):
    """ Workspace Page for models. It includes 5 options:
            1. Train/Fine-Tune models.
            2. Evaluate/Calibrate models.
            3. Interpret/Explain models.
            4. Train betting agents.
            5. Manage models & agents.

        Attributes
        ----------
        :param settings: Dict[str, Any] application settings.
        :param parent: (QWidget | None) parent widget.
    """

    def __init__(self, settings: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(
            title='Models',
            description='Build, evaluate, interpret, and manage Machine Learning models and optimal betting agents.',
            parent=parent,
        )

        self._settings = settings
        self._model_dialogs = {
            'Decision Tree': ('src.gui.components.dialogs.models.train.classifiers.tree', 'DecisionTreeTrainerDialog'),
            'Discriminant Analysis': ('src.gui.components.dialogs.models.train.classifiers.lda', 'DiscriminantAnalysisTrainerDialog'),
            'LightGBM': ('src.gui.components.dialogs.models.train.classifiers.gbm', 'LightGBMTrainerDialog'),
            'Logistic Regression': ('src.gui.components.dialogs.models.train.classifiers.logistic', 'LogisticRegressionTrainerDialog'),
            'K-Nearest Neighbors': ('src.gui.components.dialogs.models.train.classifiers.knn', 'KNearestNeighborsTrainerDialog'),
            'Naive Bayes': ('src.gui.components.dialogs.models.train.classifiers.bayes', 'NaiveBayesTrainerDialog'),
            'Neural Network': ('src.gui.components.dialogs.models.train.classifiers.nn', 'NeuralNetworkTrainerDialog'),
            'Random Forest': ('src.gui.components.dialogs.models.train.classifiers.forest', 'RandomForestTrainerDialog'),
            'Support Vector Machine': ('src.gui.components.dialogs.models.train.classifiers.svm', 'SupportVectorMachineTrainerDialog'),
            'XGBoost': ('src.gui.components.dialogs.models.train.classifiers.xgb', 'XGBoostTrainerDialog'),
        }
        self._agents_dialogs = {
            'Quantile Regression Deep Q-Learning (QR-DQN)': (
                'src.gui.components.dialogs.models.train.agents.qrdqn',
                'QRDQNTrainerDialog',
            ),
            'Proximal Policy Optimization (PPO)': (
                'src.gui.components.dialogs.models.train.agents.ppo',
                'PPOTrainerDialog',
            ),
        }

        grid = QGridLayout()
        grid.setSpacing(12)

        train = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/models-train.png',
            title='Train Models',
            description='Train ML models to predict match outcomes.',
            icon_size=115
        )
        train.clicked.connect(self.open_classifier_trainer_dialog)
        evaluate = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/models-evaluation.png',
            title='Evaluate Models',
            description='Evaluate the performance of trained models and agents.',
            icon_size=115,
        )
        evaluate.clicked.connect(self.open_evaluate_dialog)
        explain = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/models-explainability.png',
            title='Explain Models',
            description='Investigate how trained models generate decisions',
            icon_size=115,
        )
        explain.clicked.connect(self.open_explainer_dialog)
        train_agents_card = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/models-agents.png',
            title='Train Agents',
            description='Employ DRL Agents to discover optimal betting strategies.',
            icon_size=115,
        )
        train_agents_card.clicked.connect(self.open_agent_trainer_dialog)
        manage = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/models-manage.png',
            title='Manage Models & Agents',
            description='Review training and evaluation summaries or delete models and agents.',
            icon_size=115,
        )
        manage.clicked.connect(self.open_manage_models_dialog)
        grid.addWidget(train, 0, 0)
        grid.addWidget(train_agents_card, 0, 1)
        grid.addWidget(evaluate, 1, 0)
        grid.addWidget(explain, 1, 1)
        grid.addWidget(manage, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self.main_layout.addLayout(grid, 1)

    def open_classifier_trainer_dialog(self) -> None:
        """ Opens the classifier selector used to begin model training. """

        from src.gui.components.dialogs.models.train.trainer import ClassifierTrainerDialog

        dialog = ClassifierTrainerDialog(parent=self.window())
        dialog.model_selected.connect(self._open_classifier_training_dialog)
        dialog.exec()
        dialog.deleteLater()

    def _open_classifier_training_dialog(self, model_name: str) -> None:
        """ Opens the selected classifier's training dialog. """

        main_window = self.window()

        if model_name == 'LightGBM':
            from src.core.models.classifiers import LightGBM

            if LightGBM is None:
                QMessageBox.critical(
                    main_window,
                    'Classifier Unavailable',
                    'The LightGBM classifier is unavailable. It requires the lightgbm package to be installed.',
                )
                return

        league_config = getattr(main_window, '_league_config', None)
        df = getattr(main_window, '_df', None)
        model_database = getattr(main_window, 'model_database', None)

        if model_database is None:
            QMessageBox.warning(main_window, 'No League Loaded', 'Load a league before training a model.')
            return

        module_name, class_name = self._model_dialogs[model_name]
        try:
            trainer_dialog_cls = getattr(import_module(module_name), class_name)
        except ImportError as error:
            QMessageBox.critical(
                main_window,
                'Classifier Unavailable',
                f'{model_name} could not be loaded:\n{error}',
            )
            return

        dialog = trainer_dialog_cls(database=model_database, league_config=league_config, df=df, parent=main_window)
        dialog.exec()
        dialog.deleteLater()

    def open_agent_trainer_dialog(self) -> None:
        """ Opens the betting-agent selector used to begin agent training. """

        from src.gui.components.dialogs.models.train.trainer import AgentTrainerDialog

        dialog = AgentTrainerDialog(parent=self.window())
        dialog.agent_selected.connect(self._open_agent_training_dialog)
        dialog.exec()
        dialog.deleteLater()

    def open_manage_models_dialog(self) -> None:
        """Open the report and deletion manager for the active league."""

        main_window = self.window()
        model_database = getattr(main_window, 'model_database', None)
        agent_database = getattr(main_window, 'agent_database', None)

        if model_database is None or agent_database is None:
            QMessageBox.warning(main_window, 'No League Loaded', 'Load a league before managing models and agents.')
            return

        from src.gui.components.dialogs.models.manage import ManageModels

        dialog = ManageModels(
            model_database=model_database,
            agent_database=agent_database,
            parent=main_window,
        )
        dialog.exec()
        dialog.deleteLater()

    def open_evaluate_dialog(self) -> None:
        """Open model and betting-agent evaluation for the active league."""

        main_window = self.window()
        df = getattr(main_window, '_df', None)
        model_database = getattr(main_window, 'model_database', None)
        agent_database = getattr(main_window, 'agent_database', None)

        if df is None or model_database is None or agent_database is None:
            QMessageBox.warning(main_window, 'No League Loaded', 'Load a league before evaluating models.')
            return

        # Keep the feature import behind its action so opening the application
        # does not load evaluation-only model and agent dependencies.
        from src.gui.components.dialogs.models.evaluate import EvaluateModelDialog

        dialog = EvaluateModelDialog(
            df=df,
            model_database=model_database,
            agent_database=agent_database,
            parent=main_window,
        )
        dialog.exec()
        dialog.deleteLater()

    def open_explainer_dialog(self) -> None:
        """Open the model selector used to begin model interpretation."""

        main_window = self.window()
        df = getattr(main_window, '_df', None)
        model_database = getattr(main_window, 'model_database', None)

        if df is None or model_database is None:
            QMessageBox.warning(main_window, 'No League Loaded', 'Load a league before explaining models.')
            return

        # Keep interpretability-only dependencies behind their workspace action.
        from src.gui.components.dialogs.models.interpretability.explainer import ExplainerMenuDialog

        dialog = ExplainerMenuDialog(
            model_database=model_database,
            df=df,
            parent=main_window,
        )
        dialog.exec()
        dialog.deleteLater()

    def _open_agent_training_dialog(self, agent_name: str) -> None:
        """ Opens the selected agent's training dialog. """

        main_window = self.window()
        league_config = getattr(main_window, '_league_config', None)
        df = getattr(main_window, '_df', None)
        agent_database = getattr(main_window, 'agent_database', None)

        if agent_database is None:
            QMessageBox.warning(main_window, 'No League Loaded', 'Load a league before training an agent.')
            return

        module_name, class_name = self._agents_dialogs[agent_name]
        try:
            trainer_dialog_cls = getattr(import_module(module_name), class_name)
        except ImportError as error:
            QMessageBox.critical(
                main_window,
                'Agent Unavailable',
                f'{agent_name} could not be loaded:\n{error}',
            )
            return

        dialog = trainer_dialog_cls(database=agent_database, league_config=league_config, df=df, parent=main_window)
        dialog.exec()
        dialog.deleteLater()
