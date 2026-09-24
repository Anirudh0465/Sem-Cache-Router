# Import every module.
#
# Holds one parametrised test that imports each module in the project.
#
# Exists so CI is green from the first commit, and so a broken import or an
# accidental import time side effect is caught by the test suite rather than at
# container start.

from __future__ import annotations

import importlib
import pkgutil

import pytest

PACKAGES = ["app", "benchmarks", "evaluation"]


def _discover() -> list[str]:
    """Walk the package tree and return every importable module name.

    Discovered rather than hardcoded on purpose. A hardcoded list goes stale
    the moment someone adds a file, and it goes stale silently, which is the
    exact failure this test exists to catch.
    """
    found: list[str] = []
    for name in PACKAGES:
        package = importlib.import_module(name)
        found.append(name)
        found.extend(
            module.name
            for module in pkgutil.walk_packages(package.__path__, prefix=f"{name}.")
        )
    return sorted(found)


MODULES = _discover()


def test_discovery_found_the_tree() -> None:
    """Guard the guard: an empty or tiny module list would make this file a no-op."""
    assert len(MODULES) >= 25, f"only discovered {len(MODULES)} modules: {MODULES}"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    """Every module imports cleanly."""
    importlib.import_module(name)


def test_import_has_no_side_effects() -> None:
    """Importing the app must not open a connection.

    A positive assertion that the lifespan owns all wiring. Trivially true while
    app.main is empty, and the thing that catches us the moment it is not.
    """
    main = importlib.import_module("app.main")

    app = getattr(main, "app", None)
    if app is None:
        pytest.skip("app.main does not define an app yet")

    state = getattr(app, "state", None)
    for attribute in ("redis", "chroma", "provider", "cache", "embedder"):
        assert getattr(state, attribute, None) is None, (
            f"app.state.{attribute} was set at import time; wiring belongs in the lifespan"
        )
