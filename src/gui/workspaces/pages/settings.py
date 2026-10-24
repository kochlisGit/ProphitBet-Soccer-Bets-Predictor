from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget
)
from src.gui.components.workspaces.page import WorkspacePage
from src.gui.widgets import AnimatedToggleSwitch, IconLabel


class SettingsSection(QFrame):
    """ Collapsible panel used inside the Settings workspace.

        Attributes
        ----------
        :param title: (str) title of the section.
        :param description: (str) description of the section.
        :param icon_filepath: (str) filepath of the icon.
        :param expanded: (bool) whether the section starts as expanded.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            title: str,
            description: str,
            icon_filepath: Optional[str] = None,
            expanded: bool = True,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._expanded = expanded

        self.setObjectName('settingsSection')
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 14, 14, 14)
        self._layout.setSpacing(8)


        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        if icon_filepath is not None:
            header_layout.addWidget(IconLabel(filepath=icon_filepath, size=90))

        # Title + Description
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        title_label = QLabel(title)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)

        description_label = QLabel(description)
        font = description_label.font()
        font.setPointSize(12)
        description_label.setFont(font)
        description_label.setWordWrap(True)

        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)

        header_layout.addLayout(text_layout, 1)

        self._toggle_button = QToolButton()
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle_button.setAutoRaise(True)
        self._toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_button.clicked.connect(self.toggle)
        header_layout.addWidget(self._toggle_button)
        self._layout.addWidget(header)

        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setFrameShadow(QFrame.Shadow.Sunken)
        self._layout.addWidget(self._separator)

        self._content = QWidget()
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(10)
        self._layout.addWidget(self._content)

        self._content.setVisible(expanded)
        self._separator.setVisible(expanded)

    def toggle(self):
        self.setExpanded(not self._expanded)

    def setExpanded(self, expanded: bool):
        self._expanded = expanded

        self._content.setVisible(expanded)
        self._separator.setVisible(expanded)
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def isExpanded(self) -> bool:
        return self._expanded


class ThemeCard(QPushButton):
    """ Theme preview card. The card itself is deliberately not checkable. Only the radio button owns
        selection state, preventing generic ``QPushButton:checked`` theme rules from painting the whole theme card
        with the accent color.

        Attributes
        ----------
        :param icon_filepath: (str) filepath of the icon.
        :param name: (str) title of the theme.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            icon_filepath: str,
            name: str,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._theme_selected = False

        self.setText('')
        self.setCheckable(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(175, 145)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)

        preview = IconLabel(filepath=icon_filepath, size=(180, 98))
        layout.addWidget(preview, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Only this radio indicates the selected theme.
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)
        bottom.addStretch(1)

        self._radio = QRadioButton()
        self._radio.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        bottom.addWidget(self._radio)

        title_label = QLabel(name)
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        bottom.addWidget(title_label)
        bottom.addStretch(1)
        layout.addLayout(bottom)

        self._radio.toggled.connect(self._set_selection_visual)

        # Clicking anywhere on the preview card selects its radio button.
        self.clicked.connect(lambda: self._radio.setChecked(True))

    def _set_selection_visual(self, selected: bool):
        self._theme_selected = selected
        self.update()

    @property
    def radio(self) -> QRadioButton:
        return self._radio

    def isChecked(self) -> bool:
        return self._radio.isChecked()

    def setChecked(self, checked: bool):
        self._radio.setChecked(checked)

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._theme_selected:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(72, 174, 235, 42))


class SettingsWorkspace(WorkspacePage):
    """ Workspace Page for Settings.

        Attributes
        ----------
        :param parent: (QWidget | None) parent widget.
    """

    def __init__(self, settings: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(
            title='Settings',
            description='Configure application preferences and behavior.',
            parent=parent,
        )

        self._settings = settings

        self._system_themes = {
            'White': 'storage/icons/menu/themes-white.png',
            'Dark': 'storage/icons/menu/themes-dark.png',
            'Default': 'storage/icons/menu/themes-system.png'
        }

        self._theme_group = QButtonGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_cards = {}

        # --- Setting 1: Application Theme ---
        # Each setting is a self-contained StyledPanel. Additional settings can later be appended to self.main_layout.
        user_theme = settings['settings']['theme']
        theme_section = SettingsSection(
            title='Application Theme',
            description='Choose and preview the visual theme of ProphitBet.',
            icon_filepath='storage/icons/menu/themes-palette.png',
            expanded=True
        )
        section_layout = theme_section.content_layout

        # Adding System themes.
        system_label = QLabel('System Themes')
        system_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = system_label.font()
        font.setPointSize(14)
        font.setBold(True)
        system_label.setFont(font)
        section_layout.addWidget(system_label)

        system_grid = QGridLayout()
        system_grid.setContentsMargins(0, 0, 0, 0)
        system_grid.setHorizontalSpacing(10)
        system_grid.setVerticalSpacing(10)

        for col, (name, icon_filepath) in enumerate(self._system_themes.items()):
            button = self._create_theme_button(icon_filepath=icon_filepath, name=name)
            system_grid.addWidget(button, 0, col)
            system_grid.setColumnStretch(col, 1)

            if name == user_theme:
                button.setChecked(checked=True)

        section_layout.addLayout(system_grid)
        self.main_layout.addWidget(theme_section)

        self._set_current_theme(name=user_theme)

        # --- Setting 2: Startup ---
        startup_section = SettingsSection(
            title='Startup',
            description='Configure application startup behavior.',
            expanded=True
        )
        section_layout = startup_section.content_layout
        load_last_league = settings['settings'].get('load_last_league', False)

        load_league_layout = QHBoxLayout()
        load_league_layout.setContentsMargins(8, 4, 8, 4)
        load_league_layout.setSpacing(12)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        title_label = QLabel('Load Last League')
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)

        description_label = QLabel('Automatically load the last used league on startup.')
        font = description_label.font()
        font.setPointSize(11)
        description_label.setFont(font)

        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)

        load_last_league_toggle = AnimatedToggleSwitch(checked=load_last_league)
        load_last_league_toggle.setChecked(load_last_league)
        load_last_league_toggle.toggled.connect(self._set_load_last_league)

        load_league_layout.addLayout(text_layout)
        load_league_layout.addStretch(1)
        load_league_layout.addWidget(load_last_league_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)

        section_layout.addLayout(load_league_layout)
        self.main_layout.addWidget(startup_section)

        self.main_layout.addStretch(1)

    def _create_theme_button(self, icon_filepath: str, name: str) -> ThemeCard:
        button = ThemeCard(icon_filepath=icon_filepath, name=name)
        self._theme_group.addButton(button.radio)
        self._theme_cards[name] = button
        button.clicked.connect(lambda checked=False, theme_name=name: self._set_current_theme(theme_name))
        return button

    def _set_current_theme(self, name: str):
        button = self._theme_cards.get(name)
        self._settings['settings']['theme'] = name
        button.setChecked(True)

        style_hints = QGuiApplication.styleHints()

        if name == 'White':
            style_hints.setColorScheme(Qt.ColorScheme.Light)
        elif name == 'Dark':
            style_hints.setColorScheme(Qt.ColorScheme.Dark)
        elif name == 'Default':
            style_hints.unsetColorScheme()
        else:
            raise NotImplementedError(f'Not implemented system theme: "{name}"')

    def _set_load_last_league(self, enabled: bool):
        self._settings['settings']['load_last_league'] = enabled
