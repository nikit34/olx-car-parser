"""Dialect-portable access to keys inside the ``llm_extras`` JSON column.

SQLite reaches into the blob with ``json_extract(col, '$.key')``; PostgreSQL
casts to ``jsonb`` and uses ``->>``. Both are emitted from the same construct
so query code stays engine-agnostic. ``numeric=True`` returns a comparable
number instead of text, which PostgreSQL needs and SQLite gives for free.
"""

import re

from sqlalchemy import Float, String
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ColumnElement

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class json_field(ColumnElement):
    inherit_cache = True

    def __init__(self, col, key: str, numeric: bool = False):
        if key.startswith("$."):
            key = key[2:]
        if not _KEY_RE.match(key):
            raise ValueError(f"unsupported llm_extras key: {key!r}")
        self.col = col
        self.key = key
        self.numeric = numeric
        self.type = Float() if numeric else String()


@compiles(json_field, "sqlite")
def _compile_sqlite(element, compiler, **kw):
    return f"json_extract({compiler.process(element.col, **kw)}, '$.{element.key}')"


@compiles(json_field)
def _compile_default(element, compiler, **kw):
    expr = f"({compiler.process(element.col, **kw)})::jsonb ->> '{element.key}'"
    return f"({expr})::numeric" if element.numeric else expr


def json_sql(engine, column: str, key: str, numeric: bool = False) -> str:
    """The same accessor as :class:`json_field`, for hand-written SQL strings."""
    if key.startswith("$."):
        key = key[2:]
    if not _KEY_RE.match(key):
        raise ValueError(f"unsupported llm_extras key: {key!r}")
    if engine.dialect.name == "sqlite":
        expr = f"json_extract({column}, '$.{key}')"
        return f"CAST({expr} AS REAL)" if numeric else expr
    expr = f"({column})::jsonb ->> '{key}'"
    return f"({expr})::numeric" if numeric else expr
