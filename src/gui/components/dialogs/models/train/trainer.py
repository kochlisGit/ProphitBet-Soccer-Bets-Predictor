from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget
)
from src.gui.widgets import IconLabel


class _ModelOption(QPushButton):
    """Direct-action model or agent row containing an icon and name."""

    def __init__(self, name: str, icon_path: str, icon_size: int = 76, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        self.setText('')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 14, 6)
        layout.setSpacing(10)
        layout.addWidget(IconLabel(filepath=icon_path, size=icon_size))

        label = QLabel(name, self)
        font = label.font()
        font.setPointSize(12)
        font.setBold(True)
        label.setFont(font)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(label, 1)


class ClassifierTrainerDialog(QDialog):
    """Application-modal selector for the classifier training workflow."""

    model_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_model: Optional[str] = None
        self._options: dict[str, _ModelOption] = {}

        self._trainer_options = (
            ('Logistic Regression', 'storage/icons/classifiers/logistic-regression.png'),
            ('Naive Bayes', 'storage/icons/classifiers/naive-bayes.png'),
            ('K-Nearest Neighbors', 'storage/icons/classifiers/knn.png'),
            ('Discriminant Analysis', 'storage/icons/classifiers/lda.png'),
            ('Decision Tree', 'storage/icons/classifiers/decision-tree.png'),
            ('Random Forest', 'storage/icons/classifiers/random-forest.png'),
            ('XGBoost', 'storage/icons/classifiers/xgboost.png'),
            ('LightGBM', 'storage/icons/classifiers/lightgbm.png'),
            ('Support Vector Machine', 'storage/icons/classifiers/svm.png'),
            ('Neural Network', 'storage/icons/classifiers/neural-network.png'),
        )

        flags = Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setWindowTitle('Train Model')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.resize(850, 600)

        self._build_ui()

    @property
    def selected_model(self) -> Optional[str]:
        return self._selected_model

    @property
    def options(self) -> dict[str, _ModelOption]:
        return dict(self._options)

    def _build_ui(self) -> None:
        title = QLabel('Model Trainer', self)
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)

        description = QLabel('Choose the Machine Learning classifier you want to train.', self)
        description.setWordWrap(True)

        content = QWidget(self)
        options_layout = QGridLayout(content)
        options_layout.setContentsMargins(8, 8, 8, 8)
        options_layout.setHorizontalSpacing(12)
        options_layout.setVerticalSpacing(8)

        for index, (name, icon_path) in enumerate(self._trainer_options):
            option = _ModelOption(name, icon_path, parent=content)
            option.clicked.connect(lambda _checked=False, model_name=name: self._select_model(model_name))
            self._options[name] = option
            options_layout.addWidget(option, index // 2, index % 2)

        options_layout.setColumnStretch(0, 1)
        options_layout.setColumnStretch(1, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(scroll, 1)

    def _select_model(self, model_name: str) -> None:
        self._selected_model = model_name
        self.accept()
        self.model_selected.emit(model_name)


class AgentTrainerDialog(QDialog):
    """Application-modal selector for the betting-agent training workflow."""

    agent_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_agent: Optional[str] = None
        self._options: dict[str, _ModelOption] = {}
        self._trainer_options = (
            ('Quantile Regression Deep Q-Learning (QR-DQN)', 'storage/icons/agents/qr-dqn.png'),
            ('Proximal Policy Optimization (PPO)', 'storage/icons/agents/ppo.png'),
        )

        flags = Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setWindowTitle('Train Agent')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)

        self._build_ui()

    @property
    def selected_agent(self) -> Optional[str]:
        return self._selected_agent

    @property
    def options(self) -> dict[str, _ModelOption]:
        return dict(self._options)

    def _build_ui(self) -> None:
        title = QLabel('Agent Trainer', self)
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)

        description = QLabel('Choose the Reinforcement Learning agent you want to train.', self)
        description.setWordWrap(True)

        content = QWidget(self)
        options_layout = QGridLayout(content)
        options_layout.setContentsMargins(8, 8, 8, 8)
        options_layout.setHorizontalSpacing(0)
        options_layout.setVerticalSpacing(8)

        for index, (name, icon_path) in enumerate(self._trainer_options):
            option = _ModelOption(name, icon_path, icon_size=152, parent=content)
            option.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            option.clicked.connect(lambda _checked=False, agent_name=name: self._select_agent(agent_name))
            self._options[name] = option
            options_layout.addWidget(option, index, 0, 1, 2)
            options_layout.setRowStretch(index, 1)

        options_layout.setColumnStretch(0, 1)
        options_layout.setColumnStretch(1, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(scroll, 1)

    def _select_agent(self, agent_name: str) -> None:
        self._selected_agent = agent_name
        self.accept()
        self.agent_selected.emit(agent_name)
