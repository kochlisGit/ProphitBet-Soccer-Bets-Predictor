import logging
import sys
import traceback
import warnings
from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
from PyQt6.QtWidgets import QApplication
from src.gui.main import MainWindow

LOG_FILE = 'prophitbet.log'


def qt_message_handler(msg_type, context, message):
    level = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }.get(msg_type, logging.WARNING)
    loc = f"{context.file}:{context.line}" if context.file else "unknown"
    logging.getLogger('Qt').log(level, f"[{loc}] {message}")


def global_exception_hook(exc_type, exc_value, exc_tb):
    logger = logging.getLogger(__name__)
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting ProphitBet application")

    # Initializing app window.
    app = QApplication(sys.argv)

    # Create app window.
    logger.info("Creating MainWindow")
    window = MainWindow(app=app)
    window.show()
    logger.info("MainWindow shown, entering event loop")

    # Initialize the event loop.
    exit_code = app.exec()
    logger.info(f"Event loop exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stderr)
        ]
    )

    sys.excepthook = global_exception_hook
    qInstallMessageHandler(qt_message_handler)

    warnings.filterwarnings('ignore')
    main()
