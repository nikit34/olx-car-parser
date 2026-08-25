"""Text scans over title + description, resolved once at build time.

Three consumers used to run a regex over the raw listing text at read time:
the ISV import flags and the condition-NLP fault detector in
:func:`src.analytics.decision.decide`, and the salvage-phrasing hard block in
:func:`src.dashboard.data_loader._blocking_deal_reason`. That worked on the
server, where the text is a DB column — but the browser dashboard reads the
same code against ``listings.parquet``, so the whole ``description`` column had
to be shipped to the client just to be regex-scanned there.

It was 57% of the witness (15.75 MB of 27.6 MB zstd) and pushed the file past
Cloudflare's 25 MiB per-asset limit, which broke every Worker deploy from
2026-08-24. The text is not otherwise rendered: the browser needs the verdicts
of these scans, not the prose behind them.

So the scans run once, host-side, and land as four narrow columns:

  ``text_import_flag``        int8  — imported (structured ``origin`` or text)
  ``text_import_legalised``   int8  — ISV already paid / nationalised
  ``text_minor_fault``        str   — disclosed-fault label, else None
  ``text_hard_block_phrase``  str   — matched salvage phrasing, else None

Consumers read the column when it is present and fall back to scanning the raw
text when it is absent, so a CLI run against the DB, an older witness, or a
test that hands in a bare row all keep working.
"""

from __future__ import annotations

import re

import pandas as pd

# Salvage / non-runner phrases that hard-block a deal even when enrichment is
# stale (i.e. ``damage_severity`` was set under the old regex). Scans title +
# description: the 2026-05-02 audit cases JmUNP / JmutI / JmR3C all had the
# giveaway phrase only in the description, so a title-only scan would miss them.
#
# The phrase set is the union of ``llm_enrichment._PARTS_ONLY_HARD_PATTERN``,
# ``_NON_RUNNER_HARD_PATTERN``, and ``_SEVERE_DAMAGE_PATTERN``. We re-list them
# rather than import to keep this module free of the enrichment stack — it is
# mounted in the browser bundle, where that import would pull in the world.
# Staleness is acceptable because both modules are touched in lock-step.
HARD_BLOCK_TEXT_PATTERN = re.compile(
    r"para\s+pe[çc]as|vender\s+as\s+pe[çc]as|venda\s+de\s+pe[çc]as|"
    r"vende[-\s]se\s+a?\s*pe[çc]as|"
    r"para\s+sucata|para\s+desmanchar|s[óo]\s+pe[çc]as|abate|"
    r"sem\s+documentos|sem\s+matr[ií]cula|"
    r"motor\s+(?:fundido|avariad[oa])|caixa\s+avariad[oa]|"
    r"transmiss[ãa]o\s+avariad[oa]|capotamento|"
    r"avaria\s+(?:no|do)\s+motor|"
    r"junta\s+(?:de\s+cabe[çc]a\s+)?queimada|"
    r"n[ãa]o\s+pega|n[ãa]o\s+anda|n[ãa]o\s+funciona|"
    r"(?:o\s+carro\s+)?n[ãa]o\s+liga|n[ãa]o\s+arranca|"
    r"n[ãa]o\s+(?:é\s+)?poss[ií]vel\s+test(?:ar|á-lo)|"
    r"(?:s[óo]|apenas)\s+(?:de\s+|com\s+)?reboque|"
    r"non[\s-]runner|engine\s+seized",
    re.IGNORECASE,
)

# Column names, listed once so callers don't hardcode the strings.
TEXT_SIGNAL_COLUMNS = (
    "text_import_flag",
    "text_import_legalised",
    "text_minor_fault",
    "text_hard_block_phrase",
)


def _clean(value) -> str:
    """Coerce a cell to a scannable string. pandas missing values arrive as
    float NaN, and ``str(nan)`` would put the literal "nan" in the haystack."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value if isinstance(value, str) else str(value)


def hard_block_phrase(title, description) -> str | None:
    """Return the matched salvage phrasing (lowercased), else None."""
    haystack = f"{_clean(title)} {_clean(description)}"
    if not haystack.strip():
        return None
    m = HARD_BLOCK_TEXT_PATTERN.search(haystack)
    return m.group(0).lower() if m else None


def add_text_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four text-signal columns. No-op without a ``title`` column.

    Runs even when ``description`` is missing — a title-only scan is weaker
    but still correct, and it keeps the columns present (and therefore
    authoritative for consumers) on frames that never carried the prose.
    """
    if "title" not in df.columns:
        return df

    # Imported here rather than at module scope: this module is mounted in the
    # browser bundle, where consumers only touch the columns, and valuations
    # pulls in the pricing stack.
    from src.analytics.condition_signal import detect_minor_fault
    from src.analytics.valuations import _import_flags

    titles = df["title"] if "title" in df.columns else pd.Series([None] * len(df))
    descs = df["description"] if "description" in df.columns else pd.Series([None] * len(df))
    origins = df["origin"] if "origin" in df.columns else pd.Series([None] * len(df))

    imp_flags: list[int] = []
    leg_flags: list[int] = []
    faults: list[str | None] = []
    blocks: list[str | None] = []
    for title, desc, origin in zip(titles, descs, origins, strict=False):
        t, d = _clean(title), _clean(desc)
        imp, leg = _import_flags(t, d, origin if isinstance(origin, str) else None)
        imp_flags.append(int(imp))
        leg_flags.append(int(leg))
        faults.append(detect_minor_fault(t, d))
        blocks.append(hard_block_phrase(t, d))

    df["text_import_flag"] = pd.array(imp_flags, dtype="int8")
    df["text_import_legalised"] = pd.array(leg_flags, dtype="int8")
    df["text_minor_fault"] = faults
    df["text_hard_block_phrase"] = blocks
    return df
