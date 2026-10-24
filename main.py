import multiprocessing as mp
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from src.gui.main import MainMenu

    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    return app.exec()


if __name__ == '__main__':
    mp.freeze_support()
    main()
