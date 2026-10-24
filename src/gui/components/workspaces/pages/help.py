import webbrowser
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication, QGridLayout, QMessageBox, QProgressDialog, QWidget
from src.gui.components.workspaces.page import WorkspacePage, WorkspaceActionCard


class ApplicationUpdateThread(QThread):
    """Download, install, and verify an update without blocking the GUI."""

    progress = Signal(int)
    updated = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        from src.core.network.updater import update_from_github

        try:
            self.updated.emit(update_from_github(self.progress.emit))
        except Exception as error:
            self.failed.emit(str(error))


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
        update.clicked.connect(self.update_application)
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

        self._update_thread = None
        self._update_progress = None

    def update_application(self) -> None:
        """Replace all project files from GitHub main and restart the application."""

        if self._update_thread is not None and self._update_thread.isRunning():
            return

        answer = QMessageBox.question(
            self,
            'Update ProphitBet',
            'Download the latest ProphitBet version and overwrite existing '
            'application files?\n\nThe application will close and restart automatically.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._update_progress = QProgressDialog(
            'Downloading and installing all files from GitHub main...', '', 0, 100, self
        )
        self._update_progress.setWindowTitle('Updating ProphitBet')
        self._update_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._update_progress.setCancelButton(None)
        self._update_progress.setMinimumDuration(0)
        self._update_progress.setAutoClose(False)
        self._update_progress.setValue(0)

        self._update_thread = ApplicationUpdateThread(self)
        self._update_thread.progress.connect(self._update_progress.setValue)
        self._update_thread.updated.connect(self._update_completed)
        self._update_thread.failed.connect(self._update_failed)
        self._update_thread.finished.connect(self._update_finished)
        self._update_thread.finished.connect(self._update_thread.deleteLater)
        self._update_thread.start()

    def _update_completed(self, result) -> None:
        """Restart only after every downloaded file has passed verification."""

        try:
            from src.core.network.updater import restart_application

            if self._update_progress is not None:
                self._update_progress.setLabelText(
                    f'Updated and verified {result.file_count} files. Restarting ProphitBet...'
                )
                self._update_progress.setValue(100)
            restart_application(result)
        except Exception as error:
            self._restart_failed(str(error))
            return

        application = QApplication.instance()
        if application is not None:
            application.exit(0)

    def _update_failed(self, error: str) -> None:
        if self._update_progress is not None:
            self._update_progress.close()
        QMessageBox.critical(
            self,
            'Update Failed',
            f'ProphitBet could not be updated.\n\n{error}',
        )

    def _restart_failed(self, error: str) -> None:
        if self._update_progress is not None:
            self._update_progress.close()
        QMessageBox.critical(
            self,
            'Restart Failed',
            'All application files were updated and verified, but ProphitBet '
            f'could not restart automatically. Please open it manually.\n\n{error}',
        )

    def _update_finished(self) -> None:
        self._update_thread = None

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
