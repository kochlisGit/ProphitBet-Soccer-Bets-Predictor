"""Agent training dialogs loaded on first access."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    'PPOTrainerDialog': ('src.gui.components.dialogs.models.train.agents.ppo', 'PPOTrainerDialog'),
    'QRDQNTrainerDialog': ('src.gui.components.dialogs.models.train.agents.qrdqn', 'QRDQNTrainerDialog'),
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
