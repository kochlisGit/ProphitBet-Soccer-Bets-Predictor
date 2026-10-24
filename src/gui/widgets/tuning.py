from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QSizePolicy, QWidget


class TunableWidget(QWidget):
    """ A checkbox paired with a widget whose fixed value can be disabled.

        Attributes
        ----------
        :param name: (str) name of the widget.
        :param widget_id: (str) widget id.
        :param checked: (bool) whether the widget is enabled.
        :param widget: (QWidget) widget whose fixed value can be disabled.
        :param parent: (QWidget | None) parent's widget (optional).
    """

    def __init__(
            self,
            name: str,
            widget_id: str,
            checked: bool,
            widget: QWidget,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._widget_id = widget_id
        self._widget = widget
        self._checkbox = QCheckBox(name, self)
        self._checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._checkbox)
        layout.addWidget(self._widget)
        layout.addStretch(1)

        self._checkbox.toggled.connect(self._widget.setDisabled)
        self._checkbox.setChecked(checked)
        self._widget.setDisabled(checked)

    @property
    def checked(self) -> bool:
        """ Returns whether tuning is enabled and the paired widget is disabled. """

        return self._checkbox.isChecked()

    @property
    def widget_id(self) -> str:
        """ Return sthe model hyperparameter name associated with this control. """

        return self._widget_id

    @property
    def widget(self) -> QWidget:
        """ Returns the wrapped manual-value widget. """

        return self._widget
