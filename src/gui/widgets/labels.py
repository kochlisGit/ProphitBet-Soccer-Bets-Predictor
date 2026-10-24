from typing import Optional, Tuple, Union
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QWidget


class IconLabel(QLabel):
    """ Label widget that displays an icon (or an image) on the left.

        Attributes
        ----------
        :param filepath: (str) filepath of the icon (or image).
        :param size: (int) size of the icon.
        :param parent: (QWidget | None) parent of the label.
    """

    def __init__(self, filepath: str, size: Union[int, Tuple[int, int]], parent: Optional[QWidget] = None):
        super().__init__(parent)

        if isinstance(size, int):
            width = height = size
        else:
            width, height = size

        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap(filepath).scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
