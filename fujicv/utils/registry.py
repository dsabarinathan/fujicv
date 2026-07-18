"""Generic registry for losses, metrics, and other components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """A generic registry that maps string names to classes or functions.

    Usage::

        MY_REGISTRY = Registry("my_registry")

        @MY_REGISTRY.register("my_class")
        class MyClass:
            ...

        obj = MY_REGISTRY.get("my_class")()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class or function under *name*.

        If *name* is omitted the class/function ``__name__`` is used.
        """

        def decorator(obj: Any) -> Any:
            key = name if name is not None else obj.__name__
            if key in self._registry:
                raise KeyError(
                    f"Registry '{self._name}' already contains an entry for '{key}'. "
                    "Use a different name or remove the existing entry."
                )
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """Return the registered object for *name*, raising ``KeyError`` if missing."""
        if name not in self._registry:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"'{name}' is not registered in '{self._name}'. "
                f"Available: {available}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={sorted(self._registry.keys())})"
