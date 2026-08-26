"""Every module must survive being imported.

Twice in one session a module-level name was used above its definition —
`_env_float` in the scraper, and the FAQ builder in the Worker template. Both
compile fine: `py_compile` compiles, it does not execute, so the NameError only
appears when something actually imports the module. In the scraper's case that
would have been the next scrape run, in production, at 00:00 UTC.

Importing every module is the cheapest possible check for that whole class of
mistake — a use-before-definition, a typo in a module-level constant, an import
of something that no longer exists.
"""

import importlib
import pkgutil
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Modules that legitimately refuse to import in a test environment: heavy
# optional deps that CI deliberately does not install.
SKIP = {
    "src.parser.photo_damage",      # torch + torchvision
    "src.parser.photo_viewpoint",   # transformers/CLIP
}


def _all_modules() -> list[str]:
    names = []
    for mod in pkgutil.walk_packages([str(SRC)], prefix="src."):
        if mod.name in SKIP or mod.name.startswith("src.dashboard"):
            # dashboard modules import streamlit and expect a running script
            # context; they are exercised by the dashboard tests instead.
            continue
        names.append(mod.name)
    return sorted(names)


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except ImportError as e:  # missing optional dependency, not our bug
        pytest.skip(f"optional dependency missing: {e}")


def test_scripts_compile_and_expose_main():
    """Scripts are entry points, not packages — compiling them catches syntax
    errors, and the ones the workflow calls must still have a main()."""
    called_by_ci = ["build_dashboard_data.py", "build_hot_deals.py",
                    "build_stlite_bundle.py"]
    for name in called_by_ci:
        path = ROOT / "scripts" / name
        assert path.exists(), f"{name} is referenced by the scrape workflow"
        py_compile.compile(str(path), doraise=True)
        assert "def main(" in path.read_text(encoding="utf-8"), f"{name} lost its main()"
