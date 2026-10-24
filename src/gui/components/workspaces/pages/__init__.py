"""Workspace exports loaded on first access."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    'DataAnalysisWorkspace': ('src.gui.components.workspaces.pages.analysis', 'DataAnalysisWorkspace'),
    'HelpWorkspace': ('src.gui.components.workspaces.pages.help', 'HelpWorkspace'),
    'LeaguesWorkspace': ('src.gui.components.workspaces.pages.leagues', 'LeaguesWorkspace'),
    'ModelsWorkspace': ('src.gui.components.workspaces.pages.models', 'ModelsWorkspace'),
    'PredictionsWorkspace': ('src.gui.components.workspaces.pages.predictions', 'PredictionsWorkspace'),
    'SettingsWorkspace': ('src.gui.components.workspaces.pages.settings', 'SettingsWorkspace'),
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
