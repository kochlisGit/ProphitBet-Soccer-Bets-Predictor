from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QGridLayout, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from src.gui.widgets import IconLabel


class WorkspaceHeader(QWidget):
    """ Workspace header widget that displays information about the workspace page.

        Attributes
        ----------
        :param title: (str) title of the workspace.
        :param description: (str | None) description of the workspace.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(self, title: str, description: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)

        if description is None:
            description = ''

        # Adding a vertical-box layout to align title & description vertically.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # Adding title and description labels.
        title_label = QLabel(title)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        description_label = QLabel(description)
        font = description_label.font()
        font.setPointSize(11)
        description_label.setFont(font)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)


class WorkspaceActionCard(QPushButton):
    """ Workspace action card that provides users with a functionality. Each card must have the following items:
            * Icon
            * Title
            * Description

        Attributes
        ----------
        :param icon_filepath: (str) filepath of the icon.
        :param title: (str) title of the workspace.
        :param description: (str | None) description of the workspace.
        :param icon_size: (int) size of the icon.
        :param arrow: (bool) whether the title include an entry arrow.
        :param parent: (QWidget | None) parent of the workspace.
    """

    def __init__(
            self,
            icon_filepath: str,
            title: str,
            description: str,
            icon_size: int = 110,
            arrow: bool = True,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.setText('')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Adding an HBox. The PNG illustrations already contain visual whitespace --> No Additional Margin.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 14, 0)
        layout.setSpacing(0)

        icon_label = IconLabel(filepath=icon_filepath, size=icon_size)
        layout.addWidget(icon_label)

        # Adding VBox for title and description.
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addStretch(1)

        content_layout = QGridLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(6)
        content_layout.setVerticalSpacing(3)

        if arrow:
            content_layout.addWidget(
                IconLabel(filepath='storage/icons/menu/arrow.png', size=32), 0, 0, alignment=Qt.AlignmentFlag.AlignVCenter
            )

        title_label = QLabel(title)
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        content_layout.addWidget(title_label, 0, 1, alignment=Qt.AlignmentFlag.AlignVCenter)

        description_label = QLabel(description)
        font = description_label.font()
        font.setPointSize(11)
        description_label.setFont(font)
        description_label.setWordWrap(True)
        description_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        content_layout.addWidget(description_label, 1, 1)
        content_layout.setColumnStretch(1, 1)

        center_layout.addLayout(content_layout)
        center_layout.addStretch(1)

        layout.addLayout(center_layout, 1)


class WorkspacePage(QWidget):
    """ Base Workspace page class.

            Attributes
            ----------
            :param title: (str) title of the workspace.
            :param description: (str | None) description of the workspace.
            :param parent: (QWidget | None) parent of the workspace.
        """

    def __init__(self, title: str, description: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(28, 26, 28, 26)

        # Keep the page header compact. Every workspace receives the same subtle separator so the page heading is
        # visually distinct from the actual workspace controls without introducing a large empty gap.
        self.main_layout.setSpacing(10)

        self.header = WorkspaceHeader(title=title, description=description)
        self.main_layout.addWidget(self.header)

        # Adding a horizontal workspace separator between header and the page.
        separator = QFrame()
        separator.setObjectName('workspaceSeparator')
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(separator)
