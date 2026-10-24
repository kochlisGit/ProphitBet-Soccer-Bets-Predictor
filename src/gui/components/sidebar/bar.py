from typing import List
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QVBoxLayout
from src.gui.animations.hover import OnHoverScale, OnHoverImageRotation
from src.gui.widgets.led import LedIndicator
from typing import Optional
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget


class SidebarButton(QPushButton):
    """ Sidebar Button Widget.

        Attributes
        ----------
        :param text: (str) button's text.
        :param icon_filepath: (str) filepath of icon.
        :param parent: (QWidget | None) parent's widget.
    """

    def __init__(self, text: str, icon_filepath: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(text, parent)

        self.setText('')
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 0, 2)

        # Adding icon.
        if icon_filepath is not None:
            icon_label = QLabel(self)

            pixmap = QPixmap(icon_filepath)
            pixmap = pixmap.scaled(
                QSize(40, 40),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)
            icon_label.setFixedSize(40, 40)

            # Let mouse events pass through to QPushButton.
            icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(icon_label)

        # Adding title.
        title_label = QLabel(text, self)
        font = title_label.font()
        font.setPixelSize(16)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(title_label, stretch=1)


class Sidebar(QFrame):
    """ Sidebar widget, where the user selects the desired functionality.

        Includes:
            1. Title & Icon.
            2. Buttons (Menu Items).
            3. Settings (Button)
            4. Status.
    """

    page_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Theme/default style controls the actual panel.
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(240)
        self.setMaximumWidth(285)

        # Adjusting layout margins and spaces.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Title ---

        title_layout = QHBoxLayout()
        title_layout.setSpacing(12)

        ball_image = QPixmap(
            'storage/icons/menu/ball.png'
        ).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        rotating_ball_image = OnHoverImageRotation(pixmap=ball_image, parent=self)

        title_text_layout = QVBoxLayout()
        title_text_layout.setSpacing(0)

        title = QLabel('ProphitBet')
        font = title.font()
        font.setPixelSize(16)
        font.setBold(True)
        title.setFont(font)

        subtitle = QLabel('Soccer predictions')
        font = subtitle.font()
        font.setPixelSize(16)
        subtitle.setFont(font)

        title_text_layout.addWidget(title)
        title_text_layout.addWidget(subtitle)

        # Adding icon, title to layout.
        title_layout.addWidget(rotating_ball_image)
        title_layout.addLayout(title_text_layout)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        layout.addSpacing(4)

        # --- Navigation ---

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addSpacing(4)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        # Adding sidebar items.
        items = [
            ('Leagues', 'storage/icons/menu/trophy.png'),
            ('Data Analysis', 'storage/icons/menu/barplot.png'),
            ('Models', 'storage/icons/menu/network.png'),
            ('Predictions', 'storage/icons/menu/goal.png'),
            ('Settings', 'storage/icons/menu/gear.png'),
            ('Help', 'storage/icons/menu/help.png')
        ]

        self._buttons = []
        for index, (text, icon_filepath) in enumerate(items):
            button = SidebarButton(text=text, icon_filepath=icon_filepath)
            hover_wrapper = OnHoverScale(button)
            hover_wrapper.setFixedHeight(button.minimumHeight() + 6)
            layout.addWidget(hover_wrapper)
            button.clicked.connect(lambda checked=False, i=index: self.page_requested.emit(i))
            self._button_group.addButton(button, index)
            self._buttons.append(button)

        # Start at Leagues.
        self._buttons[0].setChecked(True)
        layout.addStretch()

        # --- Status ---

        self._status = LedIndicator(ready=False, text='No League Currently Loaded')
        layout.addWidget(self._status)
        layout.addSpacing(4)

    @property
    def button_group(self) -> QButtonGroup:
        return self._button_group

    @property
    def buttons(self) -> List[SidebarButton]:
        return self._buttons

    @property
    def status(self) -> LedIndicator:
        return self._status
