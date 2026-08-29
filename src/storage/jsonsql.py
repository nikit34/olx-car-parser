"""Access to keys inside the ``llm_extras`` JSON column.

The column is TEXT, so every read casts to ``jsonb`` first. ``numeric=True``
returns a comparable number rather than text — without the cast ``->>``
yields text and a comparison against a number is either an error or a
lexical ordering.
"""

import re

from sqlalchemy import Float, String
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ColumnElement

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validated(key: str) -> str:
    if key.startswith("$."):
        key = key[2:]
    if not _KEY_RE.match(key):
        raise ValueError(f"unsupported llm_extras key: {key!r}")
    return key


class json_field(ColumnElement):
    inherit_cache = True

    def __init__(self, col, key: str, numeric: bool = False):
        self.col = col
        self.key = _validated(key)
        self.numeric = numeric
        self.type = Float() if numeric else String()


@compiles(json_field)
def _compile(element, compiler, **kw):
    expr = f"({compiler.process(element.col, **kw)})::jsonb ->> '{element.key}'"
    return f"({expr})::numeric" if element.numeric else expr


def json_sql(column: str, key: str, numeric: bool = False) -> str:
    """The same accessor as :class:`json_field`, for hand-written SQL."""
    expr = f"({column})::jsonb ->> '{_validated(key)}'"
    return f"({expr})::numeric" if numeric else expr
