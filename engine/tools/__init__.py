"""PyMordial Engine Tools and Build Utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.tools.packager import PackagingConfig, ProjectPackager

__all__ = ["PackagingConfig", "ProjectPackager"]


def __getattr__(name: str):
    if name in ("PackagingConfig", "ProjectPackager"):
        import engine.tools.packager as pkg
        return getattr(pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

