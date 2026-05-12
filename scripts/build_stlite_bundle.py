#!/usr/bin/env python3
"""Assemble the stlite bundle for Cloudflare Pages deployment.

Copies the dashboard's Python sources into ``dashboard-static/files/`` with
ASCII-safe filenames; the emoji-prefixed names live only in MEMFS (mapped
via ``index.html``'s ``files: {}`` table). CF Pages serves the directory
as static content; the browser fetches each file via stlite's mount
config, and the witness parquets stream directly from the
``latest-data`` GitHub Release at mount time.

With ``--include-data`` we also copy ``data/dashboard/*`` into the bundle
so local ``python -m http.server`` testing works without a fresh release.
``index.html`` auto-detects localhost and routes parquet URLs to the
bundled copies in that case.

Run:
    python scripts/build_stlite_bundle.py
    python scripts/build_stlite_bundle.py --include-data   # for local dev
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DASHBOARD = SRC / "dashboard"
BUNDLE_DIR = ROOT / "dashboard-static" / "files"

# Hand-curated entry points. The walker below traces every other Python
# module these reach via top-level imports. Keep the dashboard entry +
# both pages here; the AST walker handles the rest (decision, price_model,
# seller_segment, plus any future top-level imports).
ENTRYPOINTS = [
    DASHBOARD / "🔥_Recommendations.py",
    DASHBOARD / "pages" / "2_📈_Market_Direction.py",
    DASHBOARD / "pages" / "3_🔍_Model_Details.py",
]

# Map dashboard-source paths → ASCII bundle paths. The MEMFS path
# (preserved in index.html) keeps the emoji so Streamlit's sidebar
# renders the right icon; the ASCII bundle path keeps CF Pages URLs
# clean and avoids any emoji-in-URL surprises.
DASHBOARD_RENAMES = {
    DASHBOARD / "🔥_Recommendations.py":               "dashboard/Recommendations.py",
    DASHBOARD / "_cache.py":                          "dashboard/_cache.py",
    DASHBOARD / "data_loader.py":                     "dashboard/data_loader.py",
    DASHBOARD / "pages" / "2_📈_Market_Direction.py":  "dashboard/pages/2_Market_Direction.py",
    DASHBOARD / "pages" / "3_🔍_Model_Details.py":      "dashboard/pages/3_Model_Details.py",
}


def _resolve_internal(modname: str) -> Path | None:
    """Resolve a top-level dotted module name to a file in this repo.

    Supports both ``src.foo.bar`` (full package path) and bare ``foo`` if
    the file exists under ``src/dashboard/`` (the dashboard's sys.path
    insert puts that dir at the front).
    """
    parts = modname.split(".")
    candidates: list[Path] = []
    if parts[0] == "src":
        base = SRC.joinpath(*parts[1:])
        candidates += [base.with_suffix(".py"), base / "__init__.py"]
    base = DASHBOARD.joinpath(*parts)
    candidates += [base.with_suffix(".py"), base / "__init__.py"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _top_level_imports(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)) and sub is not node:
                    yield sub


def _walk_imports(start: Path, visited: set[Path]) -> None:
    if start in visited:
        return
    visited.add(start)
    tree = ast.parse(start.read_text(encoding="utf-8"))
    for node in _top_level_imports(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            if node.level and node.level > 0:
                continue  # relative imports inside src package — already resolved via files-list
            names = [node.module] if node.module else []
        for name in names:
            target = _resolve_internal(name)
            if target is not None:
                _walk_imports(target, visited)


def _bundle_path(source: Path) -> str:
    """Bundle-relative path for ``source``."""
    if source in DASHBOARD_RENAMES:
        return DASHBOARD_RENAMES[source]
    try:
        rel = source.relative_to(SRC)
        return f"src/{rel.as_posix()}"
    except ValueError:
        rel = source.relative_to(DASHBOARD)
        return f"dashboard/{rel.as_posix()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--include-data", action="store_true",
        help="Also copy data/dashboard/* into the bundle (for localhost dev).",
    )
    args = ap.parse_args()

    visited: set[Path] = set()
    for entry in ENTRYPOINTS:
        if not entry.exists():
            print(f"!! entrypoint missing: {entry}", file=sys.stderr)
            sys.exit(1)
        _walk_imports(entry, visited)

    # Also include _cache.py — imported via `from _cache import ...` in
    # pages (a bare import resolved via sys.path, which the AST walker
    # catches), but the entry file itself is captured via DASHBOARD_RENAMES.
    visited.add(DASHBOARD / "_cache.py")
    visited.add(DASHBOARD / "data_loader.py")

    # Ensure package __init__.py files come along for every src/ subpackage
    # we touch — otherwise Python can't import e.g. ``src.analytics.decision``.
    # Always include src/__init__.py (the top-level package marker); skip
    # src/dashboard/__init__.py because dashboard files are mounted at MEMFS
    # root, not under ``src.dashboard.*``.
    pkg_inits: set[Path] = set()
    if any(p.is_relative_to(SRC) and not p.is_relative_to(DASHBOARD) for p in visited):
        top_init = SRC / "__init__.py"
        if top_init.exists():
            pkg_inits.add(top_init)
    for p in visited:
        if not p.is_relative_to(SRC) or p.is_relative_to(DASHBOARD):
            continue
        cur = SRC
        rel = p.relative_to(SRC)
        for part in rel.parts[:-1]:
            cur = cur / part
            init = cur / "__init__.py"
            if init.exists():
                pkg_inits.add(init)
    visited |= pkg_inits
    # Strip anything under src/dashboard from visited — those modules live
    # at MEMFS root via DASHBOARD_RENAMES, not under src/dashboard/.
    visited = {p for p in visited if not p.is_relative_to(DASHBOARD) or p in DASHBOARD_RENAMES}

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    files_manifest: dict[str, str] = {}
    for src in sorted(visited):
        bundle_rel = _bundle_path(src)
        dst = BUNDLE_DIR / bundle_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        files_manifest[bundle_rel] = src.relative_to(ROOT).as_posix()

    print(f"[stlite] bundled {len(files_manifest)} files → {BUNDLE_DIR}")
    for k, v in sorted(files_manifest.items()):
        print(f"  {k:<45} ← {v}")

    if args.include_data:
        data_src = ROOT / "data" / "dashboard"
        if not data_src.exists():
            print(f"!! data/dashboard missing — run build_dashboard_data.py first", file=sys.stderr)
            sys.exit(2)
        data_dst = BUNDLE_DIR / "data" / "dashboard"
        if data_dst.exists():
            shutil.rmtree(data_dst)
        shutil.copytree(data_src, data_dst)
        n = sum(1 for _ in data_dst.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in data_dst.rglob("*") if f.is_file())
        print(f"[stlite] +{n} data files ({size / 1e6:.2f} MB) under data/dashboard/")


if __name__ == "__main__":
    main()
