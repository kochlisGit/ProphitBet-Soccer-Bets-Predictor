import webbrowser
from typing import Any, Dict, Optional
from PySide6.QtWidgets import QGridLayout, QMessageBox, QWidget
from src.gui.components.workspaces.page import WorkspacePage, WorkspaceActionCard


class HelpWorkspace(WorkspacePage):
    """ Workspace Page for Help. It includes 4 options:
            1. Learn (Tutorials)
            2. Update
            3. Submit Issue (to GitHub)
            4. Donate

        Attributes
        ----------
        :param parent: (QWidget | None) parent widget.
    """

    def __init__(self, settings: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(
            title='Help',
            description='Get support, learn, and contribute.',
            parent=parent
        )

        self._settings = settings

        grid = QGridLayout()
        grid.setSpacing(14)
        learn = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/help-learn.png',
            title='Learn',
            description='Learn Statistics, Machine Learning, Data Analysis, Interpretability',
            icon_size=135
        )
        learn.clicked.connect(self.open_learn_dialog)
        update = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/help-update.png',
            title='Update',
            description='Update ProphitBet to the latest version.',
            icon_size=135
        )
        bug = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/help-bug.png',
            title='Submit Bug',
            description='Open the GitHub to report a problem or a suggestion.',
            icon_size=135
        )
        bug.clicked.connect(self.open_submit_bug)
        donate = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/help-donate.png',
            title='Buy Me a Coffee :)',
            description='If you enjoy ProphitBet, consider supporting its development through PayPal.',
            icon_size=135
        )
        donate.clicked.connect(self.open_donation_page)

        grid.addWidget(learn, 0, 0)
        grid.addWidget(update, 0, 1)
        grid.addWidget(bug, 1, 0)
        grid.addWidget(donate, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self.main_layout.addLayout(grid, 1)

    def open_learn_dialog(self) -> None:
        """Open the tutorial link browser."""

        from src.gui.components.dialogs.help.learn import LearnDialog

        dialog = LearnDialog(parent=self.window())
        dialog.exec()
        dialog.deleteLater()

    def open_submit_bug(self) -> None:
        """Open the GitHub issue form in the default web browser."""

        url = 'https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/issues/new'
        if not webbrowser.open(url, new=2):
            QMessageBox.warning(
                self,
                'Browser Unavailable',
                f'Could not open the GitHub issue form:\n{url}',
            )

    def open_donation_page(self) -> None:
        """Open the PayPal donation page in the default web browser."""

        url = 'https://www.paypal.com/donate/?hosted_button_id=AK3SEFDGVAWFE'
        if not webbrowser.open(url, new=2):
            QMessageBox.warning(
                self,
                'Browser Unavailable',
                f'Could not open the PayPal donation page:\n{url}',
            )
