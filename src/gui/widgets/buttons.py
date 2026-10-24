from typing import Optional
from PySide6.QtWidgets import QPushButton, QWidget


class PrimaryQPushButton(QPushButton):
    """ QPushButton widget that has a "Primary Button" style.

        Attributes
        ----------
        :param text: (str) button's text.
        :param parent: (QWidget | None) parent of the button.
    """

    def __init__(self, text: Optional[str], parent: Optional[QWidget] = None):
        if text is None:
            text = ''

        super().__init__(text, parent)

        # Enforce name & primary styling inside the subclass
        self.setObjectName('primaryActionButton')
        self.setStyleSheet(
            """
                QPushButton#primaryActionButton {
                    background-color: #0d6efd;
                    color: white;
                    border: 1px solid #0d6efd;
                    border-radius: 8px;
                    padding: 8px 18px;
                    font-weight: bold;
                }
                QPushButton#primaryActionButton:hover {
                    background-color: #2684ff;
                    border-color: #2684ff;
                }
                QPushButton#primaryActionButton:pressed {
                    background-color: #0b5ed7;
                    border-color: #0b5ed7;
                }
            """
        )
