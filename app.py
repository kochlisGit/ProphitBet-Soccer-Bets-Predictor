import logging
import sys
import warnings
from PyQt6.QtWidgets import QApplication
from src.gui.main import MainWindow


def main():
    # Initializing app window.
    app = QApplication(sys.argv)

    # Create app window.
    window = MainWindow(app=app)
    window.show()

    # Initialize the event loop.
    app.exec()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore')
    main()
