"""Public widget exports loaded only when first requested.

Keeping this package lightweight prevents a simple widget such as
``IconLabel`` from importing the table/export stack and OpenPyXL.
"""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    'PrimaryQPushButton': ('src.gui.widgets.buttons', 'PrimaryQPushButton'),
    'CheckableComboBox': ('src.gui.widgets.combobox', 'CheckableComboBox'),
    'IconLabel': ('src.gui.widgets.labels', 'IconLabel'),
    'LedIndicator': ('src.gui.widgets.led', 'LedIndicator'),
    'ListView': ('src.gui.widgets.listview', 'ListView'),
    'CircularProgressBar': ('src.gui.widgets.progress', 'CircularProgressBar'),
    'FloatDoubleRangeSlider': ('src.gui.widgets.sliders', 'FloatDoubleRangeSlider'),
    'ExcelTable': ('src.gui.widgets.table', 'ExcelTable'),
    'AnimatedToggleSwitch': ('src.gui.widgets.toggle', 'AnimatedToggleSwitch'),
    'TunableWidget': ('src.gui.widgets.tuning', 'TunableWidget'),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
