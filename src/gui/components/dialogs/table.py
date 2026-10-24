import pandas as pd
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QPushButton, QToolBar, QVBoxLayout, QWidget
from src.gui.widgets.table import ExcelTable


class TableDialog(QDialog):
    """ Independent Table Window that shows the excel-table.

        Attributes
        ----------
        :param parent: (QWidget | None): parent of the table.
        :param title: (str) title of the table.
        :param df: (pd.DataFrame) dataframe to br displayed.
        :param readonly: (bool) whether table items can be modified.
        :param supports_sorting: (bool) whether sorting can be applied.
        :param supports_query_search: (bool) whether query search is supported.
        :param supports_deletion: (bool) whether deletion of items is supported.
        :param auto_resize: (bool) whether auto-resize of columns is enabled.
                            It can also be enabled/disabled through context menu.
        :param hide_missing_option: (bool) whether the Hide Missing Rows
                                    toolbar checkbox is available.
    """

    def __init__(
            self,
            parent: Optional[QWidget],
            title: str,
            df: pd.DataFrame,
            readonly: bool = True,
            supports_sorting: bool = True,
            supports_query_search: bool = True,
            supports_deletion: bool = True,
            auto_resize: bool = False,
            hide_missing_option: bool = True,
    ):
        super().__init__(parent, Qt.WindowType.Window)

        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)

        toolbar = QToolBar('League Data Controls', self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setMinimumHeight(56)
        toolbar.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(toolbar)

        toolbar_controls = QWidget(toolbar)
        toolbar_layout = QHBoxLayout(toolbar_controls)
        toolbar_layout.setContentsMargins(10, 6, 10, 6)
        toolbar_layout.setSpacing(12)

        self.table = ExcelTable(
            parent=self,
            df=df,
            readonly=readonly,
            supports_sorting=supports_sorting,
            supports_query_search=supports_query_search,
            supports_deletion=supports_deletion,
            auto_resize=auto_resize,
        )

        auto_resize_button = QCheckBox('Auto-Resize')
        auto_resize_button.setToolTip('Continuously resize columns to fit their contents.')
        auto_resize_button.toggled.connect(self.table.set_auto_resize)
        auto_resize_button.setChecked(auto_resize)
        toolbar_layout.addWidget(auto_resize_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        if hide_missing_option:
            hide_missing_button = QCheckBox('Hide Missing Rows')
            hide_missing_button.setToolTip('Hide rows that contain at least one missing value.')
            hide_missing_button.toggled.connect(self.table.hide_missing)
            hide_missing_button.setChecked(False)
            toolbar_layout.addWidget(hide_missing_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        find_replace_button = QPushButton('Find / Replace')
        find_replace_button.setMinimumHeight(36)
        find_replace_button.setToolTip('Find values in the league data.')
        find_replace_button.clicked.connect(self.table.open_find_dialog)
        toolbar_layout.addWidget(find_replace_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        export_table_button = QPushButton('Export Table')
        export_table_button.setMinimumHeight(36)
        export_table_button.setToolTip('Store the table data as a CSV or Excel file.')
        export_table_button.clicked.connect(self.table.export_table)
        toolbar_layout.addWidget(export_table_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        toolbar_layout.addStretch(1)
        toolbar.addWidget(toolbar_controls)

        dialog_layout.addWidget(self.table, 1)
        self.table.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
