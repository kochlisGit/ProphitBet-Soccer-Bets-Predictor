import numpy as np
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QDialog, QVBoxLayout


class PlotWindow(QDialog):
    def __init__(self, ax: Axes, parent: Optional[QDialog] = None, title='New Plot'):
        super().__init__(parent)

        if isinstance(ax, list) or isinstance(ax, tuple) or isinstance(ax, np.ndarray):
            figure = ax[0].figure
        else:
            figure = ax.figure

        # Create canvas to plot the ax figure.
        self.setWindowTitle(title)
        self.canvas = FigureCanvas(figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Draw canvas figure.
        self.canvas.draw_idle()
