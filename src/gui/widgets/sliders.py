from PyQt6.QtWidgets import QSlider


def add_snap_behavior(slider: QSlider, step):
    def snap(v):
        mn = slider.minimum()
        mx = slider.maximum()
        snapped = mn + round((v - mn) / step)*step
        snapped = max(mn, min(mx, snapped))
        if snapped != v:
            slider.blockSignals(True)
            slider.setValue(snapped)
            slider.blockSignals(False)

    slider.valueChanged.connect(snap)
