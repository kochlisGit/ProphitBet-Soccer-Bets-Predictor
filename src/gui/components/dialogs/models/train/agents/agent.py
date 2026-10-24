import pandas as pd
from typing import Any, List, Optional, Type
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget
)
from src.core.database.agent import AgentDatabase
from src.core.leagues.league import LeagueConfig
from src.core.agents.tuner import TuningAlgorithm
from src.core.agents.utils import create_agent
from src.gui.components.dialogs.models.train.agents.utils import TrainEvalReportDialog
from src.gui.components.dialogs.tasks import SnakeTaskDialog
from src.gui.widgets import PrimaryQPushButton


class AgentDialog(QDialog):
    """ Base dialog containing settings shared by all agent trainers.

        Attributes
        ----------
        :param agent_cls: (Type) trainer dialog type. Each agent has a unique dialog type.
        :param database: (AgentDatabase) active league's agent database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(
            self,
            agent_cls: Type,
            database: AgentDatabase,
            league_config: LeagueConfig,
            df: pd.DataFrame,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._agent_cls = agent_cls
        self._league_config = league_config
        self._df = df.dropna().sort_values(by='Date', ascending=False).reset_index(drop=True)

        self._agent_database = database

        self._agent_id = None
        self._test_size = None
        self._hyperparameter_widgets = None
        self._train_eval_report_dialog = None

        flags = Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setWindowTitle(f'Train {agent_cls.__name__}')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)

        self._build_ui()

        font = self.font()
        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 1)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + 1)
        self.setFont(font)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[QWidget]:
        """Add algorithm-specific hyperparameter controls to ``layout``."""

        raise NotImplementedError('All agents dialogs must implement "add_hyperparameters" method.')

    def _get_value(self, name: str) -> Any:
        """ Return the current manual value for a hyperparameter. """

        raise NotImplementedError('All agents dialogs must implement "_get_value" method.')

    def _find_available_agent_id(self) -> str:
        """Return the first available agent-specific, two-digit ID."""

        number = 1
        prefix = self._agent_cls.__name__
        while self._agent_database.agent_exists(agent_id := f'{prefix}-{number:02d}'):
            number += 1
        return agent_id

    def _build_ui(self):
        def section_title(text: str, parent: QWidget) -> QLabel:
            label = QLabel(text, parent)
            label_font = label.font()
            label_font.setPointSize(label_font.pointSize() + 2)
            label_font.setBold(True)
            label.setFont(label_font)
            label.setProperty('sectionTitle', True)
            return label

        def new_section(parent: QWidget) -> tuple[QFrame, QVBoxLayout]:
            frame = QFrame(parent)
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(14, 12, 14, 14)
            frame_layout.setSpacing(10)
            return frame, frame_layout

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(12)

        agent_frame, agent_layout = new_section(content)
        content_layout.addWidget(section_title('Agent Settings', content))

        settings_row = QHBoxLayout()
        settings_row.setSpacing(12)

        agent_id_label = QLabel('Agent ID*', agent_frame)
        self.agent_id_edit = QLineEdit(agent_frame)
        self.agent_id_edit.setClearButtonEnabled(True)
        self.agent_id_edit.setText(self._find_available_agent_id())
        self.agent_id_edit.setFixedWidth(300)
        self.agent_id_edit.setToolTip('Enter a unique agent identifier. It should begin with a letter.')

        self.invalid_id_label = QLabel(agent_frame)
        self.invalid_id_label.setWordWrap(True)
        self.invalid_id_label.setStyleSheet('QLabel { color: #E74C3C; font-weight: 600; }')
        self.invalid_id_label.hide()

        settings_row.addWidget(agent_id_label)
        settings_row.addWidget(self.agent_id_edit)
        settings_row.addWidget(self.invalid_id_label)
        settings_row.addStretch(1)
        agent_layout.addLayout(settings_row)
        content_layout.addWidget(agent_frame)

        hyperparameters_frame, hyperparameters_layout = new_section(content)
        content_layout.addWidget(section_title('Hyperparameters', content))

        instruction = QLabel('Check hyperparameters to tune them or select them manually.', hyperparameters_frame)
        hyperparameters_layout.addWidget(instruction, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,)
        self._hyperparameter_widgets = self._add_hyperparameters(
            frame=hyperparameters_frame,
            layout=hyperparameters_layout
        )

        hyperparameters_layout.activate()
        hyperparameters_frame.setFixedHeight(hyperparameters_layout.sizeHint().height())
        content_layout.addWidget(hyperparameters_frame)

        self.training_settings_frame, self.training_settings_layout = new_section(content)
        content_layout.addWidget(section_title('Training/Tuning Settings', content))
        self._build_training_settings()
        content_layout.addWidget(self.training_settings_frame)
        content_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)

        self.train_button = PrimaryQPushButton('Start Training', self)
        font = self.train_button.font()
        font.setPointSize(font.pointSize() + 1)
        self.train_button.setIcon(QIcon('storage/icons/menu/play.png'))
        self.train_button.setIconSize(QSize(32, 32))
        self.train_button.setToolTip('Start hyperparameter tuning, training, and evaluation.')
        self.train_button.setMinimumSize(120, 42)
        self.train_button.clicked.connect(self._train)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.train_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 14, 14)
        layout.addWidget(scroll)
        layout.addLayout(actions)

        self.agent_id_edit.textChanged.connect(self._validate_agent_id)
        self._validate_agent_id(self.agent_id_edit.text())

    def _build_training_settings(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.test_size_type_combo = QComboBox(self.training_settings_frame)
        self.test_size_type_combo.addItem('Num. Samples', userData=int)
        self.test_size_type_combo.addItem('Samples (%)', userData=float)
        self.test_size_type_combo.setCurrentIndex(0)
        self._test_size_type = int
        self.test_size_type_combo.setToolTip(
            'Choose whether test size is an exact sample count or a fraction of the dataset.'
        )

        self.test_size_slider = QSlider(Qt.Orientation.Horizontal, self.training_settings_frame)
        self.test_size_slider.setToolTip('Select the number or percentage of samples held out for testing.')
        self.test_size_value_label = QLabel(self.training_settings_frame)

        self._trials_slider = QSlider(Qt.Orientation.Horizontal, self.training_settings_frame)
        self._trials_slider.setRange(50, 1000)
        self._trials_slider.setSingleStep(50)
        self._trials_slider.setPageStep(50)
        self._trials_slider.setTickInterval(50)
        self._trials_slider.setValue(200)
        self._trials_slider.setMaximumWidth(180)
        self._trials_slider.setToolTip('Select the number of hyperparameter optimization trials.')
        self.trials_value_label = QLabel(str(self._trials_slider.value()), self.training_settings_frame)

        self._tuning_algorithm_combo = QComboBox(self.training_settings_frame)
        for algorithm in TuningAlgorithm:
            self._tuning_algorithm_combo.addItem(algorithm.name, userData=algorithm)
        self._tuning_algorithm_combo.setMaximumWidth(160)
        self._tuning_algorithm_combo.setToolTip('Select the algorithm used to search hyperparameters.')

        grid.addWidget(QLabel('Test Size Type', self.training_settings_frame), 0, 0)
        grid.addWidget(self.test_size_type_combo, 0, 1)
        grid.addWidget(QLabel('Test Size', self.training_settings_frame), 0, 2)
        grid.addWidget(self.test_size_slider, 0, 3)
        grid.addWidget(self.test_size_value_label, 0, 4)
        grid.addWidget(QLabel('Trials', self.training_settings_frame), 1, 0)
        grid.addWidget(self._trials_slider, 1, 1)
        grid.addWidget(self.trials_value_label, 1, 2)
        grid.addWidget(QLabel('Tuning Algorithm', self.training_settings_frame), 1, 3)
        grid.addWidget(self._tuning_algorithm_combo, 1, 4)

        self.training_settings_layout.addLayout(grid)

        self.test_size_type_combo.currentIndexChanged.connect(self._configure_test_size)
        self.test_size_slider.valueChanged.connect(self._update_test_size_label)
        self._trials_slider.valueChanged.connect(self._update_trials)
        self._configure_test_size(preserve_value=False)

    @staticmethod
    def _snap_value(value: int, minimum: int, step: int) -> int:
        return minimum + round((value - minimum) / step) * step

    def _validate_agent_id(self, text: Optional[str] = None) -> bool:
        """Validate the agent identifier and display the first error."""

        if text is None:
            text = self.agent_id_edit.text()

        self._agent_id = text.strip()

        if not self._agent_id:
            message = 'Agent ID is required.'
        elif self._agent_database.agent_exists(agent_id=self._agent_id):
            message = 'Agent ID already exists.'
        elif not self._agent_id[0].isalpha():
            message = 'Agent ID must start with a letter.'
        elif any(not (character.isalnum() or character in '_-') for character in self._agent_id):
            message = 'Only letters, numbers, "_" and "-" are allowed.'
        else:
            self.invalid_id_label.clear()
            self.invalid_id_label.hide()
            return True

        self._agent_id = None
        self.invalid_id_label.setText(message)
        self.invalid_id_label.show()
        return False

    def _configure_test_size(self, _index: int = -1, preserve_value: bool = True):
        new_type = self.test_size_type_combo.currentData()
        old_type = self._test_size_type
        old_value = self.test_size_slider.value()
        sample_count = max(len(self._df), 1)

        if new_type is int:
            maximum = max(50, (int(len(self._df) * 0.30) // 50) * 50)
            self.test_size_slider.setRange(50, maximum)
            self.test_size_slider.setSingleStep(50)
            self.test_size_slider.setPageStep(50)
            self.test_size_slider.setTickInterval(50)

            if preserve_value and old_type is float:
                converted = self._snap_value(round(sample_count * old_value / 100), 50, 50)
                value = min(maximum, max(50, converted))
            else:
                value = min(maximum, 100)
        else:
            self.test_size_slider.setRange(1, 30)
            self.test_size_slider.setSingleStep(1)
            self.test_size_slider.setPageStep(1)
            self.test_size_slider.setTickInterval(1)

            if preserve_value and old_type is int:
                value = min(30, max(1, round(old_value * 100 / sample_count)))
            else:
                value = 20

        self._test_size_type = new_type
        self.test_size_slider.setValue(value)
        self._update_test_size_label(value)

    def _update_test_size_label(self, value: int):
        if self.test_size_type_combo.currentData() is int:
            snapped = self._snap_value(value, 50, 50)

            if snapped != value:
                self.test_size_slider.setValue(snapped)
                return

            self._test_size = value
            text = f'{value} samples'
        else:
            self._test_size = value / 100
            text = f'{value}%'
        self.test_size_value_label.setText(text)

    def _update_trials(self, value: int):
        snapped = self._snap_value(value, 50, 50)

        if snapped != value:
            self._trials_slider.setValue(snapped)
            return

        self.trials_value_label.setText(str(value))

    def _train(self):
        """ Validates the training parameters and trains/tunes/evaluate the selected agent. """

        # Validating the agent identifier.
        if not self._validate_agent_id():
            return

        # Reading agent and tunable hyperparameters.
        suggested_params = self._agent_cls.get_suggested_hyperparameters()
        agent_params = {}
        tunable_params = {}

        for tunable_widget in self._hyperparameter_widgets:
            parameter_name = tunable_widget.widget_id
            if tunable_widget.checked:
                tunable_params[parameter_name] = suggested_params[parameter_name]
            else:
                agent_params[parameter_name] = self._get_value(name=parameter_name)

        if tunable_params:
            confirmation = QMessageBox(self)
            confirmation.setIcon(QMessageBox.Icon.Information)
            confirmation.setWindowTitle('Confirm Hyperparameter Tuning')
            confirmation.setText('Hyperparameter tuning might take some time.')
            proceed_button = confirmation.addButton('Proceed', QMessageBox.ButtonRole.AcceptRole)
            confirmation.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
            confirmation.setDefaultButton(proceed_button)
            confirmation.exec()

            if confirmation.clickedButton() is not proceed_button:
                return

            task_description = (
                f'Tuning Hyperparameters of {self._agent_cls.__name__}. '
                'Please wait, this operation might take some time...'
            )
        else:
            task_description = f'Training {self._agent_cls.__name__}. Please wait...'

        open_optuna_dashboard = len(tunable_params) > 0
        task = SnakeTaskDialog(
            create_agent,
            title='Agent Training',
            description=task_description,
            parent=self,
            league_id=self._league_config.league_id,
            agent_id=self._agent_id,
            df=self._df,
            test_size=self._test_size,
            agent_cls=self._agent_cls,
            agent_params=agent_params,
            tunable_params=tunable_params,
            trials=self._trials_slider.value(),
            tune_algorithm=self._tuning_algorithm_combo.currentData(),
            open_optuna_dashboard=open_optuna_dashboard
        )
        result = task.run()

        if result is not None:
            agent_config, agent = result
            self._agent_database.save_agent(agent_config=agent_config, agent=agent)

            QMessageBox.information(
                self,
                'Training Completed',
                f'Model "{self._agent_id}" was trained and saved successfully. '
                'Training-Evaluation table will load shortly and tuning dashboard will now close.',
            )

            self._train_eval_report_dialog = TrainEvalReportDialog(
                parent=self,
                agent_config=agent_config,
            )
            self._train_eval_report_dialog.show()
        else:
            if task.cancelled:
                QMessageBox.critical(
                    self,
                    'Training Aborted',
                    f'Training has been interrupted and stopped. Tuning dashboard will now close.',
                )
