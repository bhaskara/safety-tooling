# Helpers for the optional dependency groups declared in pyproject.toml ("media": librosa,
# soundfile, pydub, opencv; "gemini": google-generativeai, google-cloud-aiplatform). Modules that
# need one of those packages import it through try_import() at module scope so that
# `import safetytooling` and InferenceAPI construction work without the extra, and call require()
# at the point of use so the failure is an ImportError that names the extra to install.
"""Guarded imports for safetytooling's optional dependency groups."""

from __future__ import annotations

import importlib
from types import ModuleType

# Original import failure per module name, surfaced by require() so that a broken (rather than
# merely absent) install is still diagnosable.
_IMPORT_ERRORS: dict[str, ImportError] = {}


def try_import(module_name: str) -> ModuleType | None:
    """Import ``module_name`` if it is installed, otherwise return ``None``.

    Parameters
    ----------
    module_name : str
        Dotted module path, e.g. ``"vertexai.generative_models"``.

    Returns
    -------
    ModuleType or None
        The imported module, or ``None`` when importing it raised ``ImportError`` (including
        ``ModuleNotFoundError`` for a missing transitive dependency). The exception is recorded
        so ``require`` can report it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        _IMPORT_ERRORS[module_name] = exc
        return None


def require(module: ModuleType | None, module_name: str, extra: str) -> ModuleType:
    """Return ``module`` or raise an ``ImportError`` naming the extra that provides it.

    Parameters
    ----------
    module : ModuleType or None
        Result of ``try_import(module_name)``.
    module_name : str
        Dotted module path used in the error message.
    extra : str
        Name of the safetytooling optional-dependency group (``"media"`` or ``"gemini"``).

    Returns
    -------
    ModuleType
        ``module`` unchanged when it is not ``None``.

    Raises
    ------
    ImportError
        When ``module`` is ``None``; the message includes the install hint and the original
        import error.
    """
    if module is None:
        cause = _IMPORT_ERRORS.get(module_name)
        detail = f" (original error: {cause!r})" if cause is not None else ""
        raise ImportError(
            f"'{module_name}' is required for this feature but is not installed; it belongs to the "
            f"optional '{extra}' group: pip install 'safetytooling[{extra}]'{detail}"
        )
    return module
