"""Model-specific code named by profiles as `module:function` (or a bare module) and resolved at run time.

Profiles keep model quirks out of the stage scripts: a student that needs its adapter renamed for vLLM, a
training-time attention patch, or a custom merge names the function here instead of the scripts testing
for the model family. The module must be staged with the job (`requires` in the student profile).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def resolve(spec: str) -> Callable[..., Any] | Any:
    """`"pkg.module:attr"` -> the attribute; `"pkg.module"` -> the module."""
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Hook {spec!r} is not available in this checkout ({exc}); stage the module the profile requires"
        ) from exc
    if not attribute:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RuntimeError(f"Hook {spec!r}: {module_name} has no attribute {attribute!r}") from exc
