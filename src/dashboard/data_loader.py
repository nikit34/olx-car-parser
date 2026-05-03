"""Load data from SQLite database (local or downloaded from GitHub Releases)."""

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "olx_cars.db"

# Read-time mileage sanity ceiling — mirrors the write-time cap in
# :func:`src.parser.llm_enrichment.correct_listing_data`. Any odometer above
# this is a parse error in the LLM's km extraction (the loudest 2026-05-02
# case was Honda Civic JmuYR at 278_000_000 km, "278 mil km" mis-read).
_SANITY_MAX_MILEAGE_KM = 1_000_000


def _fuel_group(value) -> str:
    """Coarse fuel-type bucket used for filtering and chart legends.

    Lives here (not in app.py) so the multipage dashboard's segment-detail
    and market-trend pages can reuse the exact same canonicalisation as
    the deal feed without re-importing the Streamlit entrypoint.
    """
    import pandas as _pd
    if value is None or (isinstance(value, float) and _pd.isna(value)) or str(value).strip() == "":
        return "Unknown"
    fl = str(value).lower()
    if "diesel" in fl:
        return "Diesel"
    if "eléctrico" in fl or "electr" in fl:
        return "Electric"
    if "plug" in fl:
        return "Plug-in Hybrid"
    if "híbrido" in fl or "hybrid" in fl:
        return "Hybrid"
    if "gpl" in fl or "lpg" in fl:
        return "GPL"
    return "Petrol"

# Repair-cost heuristic for ``damage_severity == 2`` listings (needs repair
# OR accident history). We don't block these — they can still be flippable
# if the asking price is far enough below the GB-predicted "clean" price to
# absorb the bodywork / mechanical work on top. Without a parts/labor
# database the only honest approach is a percentage of the predicted clean
# price, with a higher pct + floor when ``mechanical_condition == "poor"``
# (engine / gearbox work runs much wider than panel paint). The 2026-05-02
# audit Citroën C5 8Q0kOc — starter + EGR + MAF + water leak stacked up to
# €2.5–4k easily, so the "poor" branch needs to be conservative.
_REPAIR_COST_PCT_DEFAULT = 0.12
_REPAIR_COST_PCT_POOR = 0.18
_REPAIR_COST_FLOOR_DEFAULT = 1000.0
_REPAIR_COST_FLOOR_POOR = 1500.0


def _estimate_repair_cost(
    severity: int | None,
    mech_condition: str | None,
    predicted_price: float,
) -> float:
    """Heuristic repair cost (€) for damage-severity-2 listings.

    Returns 0 for pristine / normal-wear / unknown-severity rows so the
    caller can subtract unconditionally. Severity 3 is hard-blocked
    upstream; if it ever reaches here we return ``predicted_price`` so any
    profit calc collapses to <=0 and the listing drops out.
    """
    if severity is None or severity < 2:
        return 0.0
    if severity >= 3:
        return predicted_price
    poor = (mech_condition or "").strip().lower() == "poor"
    pct = _REPAIR_COST_PCT_POOR if poor else _REPAIR_COST_PCT_DEFAULT
    floor = _REPAIR_COST_FLOOR_POOR if poor else _REPAIR_COST_FLOOR_DEFAULT
    return max(predicted_price * pct, floor)


_CHECK_INTERVAL_SECONDS = 2 * 3600  # check for new release every 2 hours


def _github_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        import streamlit as st
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


_LAST_RELEASE_ERROR: str | None = None


def get_last_release_error() -> str | None:
    """Surface the most recent release-fetch failure to the empty-state UI.

    Without this the dashboard says "No data yet" with zero hint why —
    rate limit, network, missing release, etc. all look identical to the
    user."""
    return _LAST_RELEASE_ERROR


def _list_release_assets(repo: str) -> dict[str, dict] | None:
    """Return {asset_name: asset_dict} for the latest-data release.

    On failure stamps ``_LAST_RELEASE_ERROR`` with a human-readable
    reason (HTTP status / exception class) so the empty-state banner
    can explain what's going on.
    """
    global _LAST_RELEASE_ERROR
    import httpx

    api_url = f"https://api.github.com/repos/{repo}/releases/tags/latest-data"
    headers = {"Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(api_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            hint = ""
            if resp.status_code in (403, 429):
                hint = (
                    " (likely rate-limited — set GITHUB_TOKEN in env or "
                    "Streamlit secrets to lift the 60-req/hour cap)"
                )
            _LAST_RELEASE_ERROR = (
                f"GitHub API returned HTTP {resp.status_code} for "
                f"releases/tags/latest-data{hint}"
            )
            return None
        _LAST_RELEASE_ERROR = None
        return {a["name"]: a for a in resp.json().get("assets", [])}
    except Exception as e:
        _LAST_RELEASE_ERROR = f"GitHub API call failed: {type(e).__name__}: {e}"
        return None


def _public_download_url(repo: str, asset_name: str) -> str:
    """CDN-backed direct URL for a public release asset.

    Bypasses the GitHub API rate limit entirely — works even when
    ``_list_release_assets`` returned None due to a 403/429. Costs us
    the ``updated_at`` comparison (we always download on this path),
    but on Streamlit Cloud cold-starts the local file doesn't exist
    anyway so the comparison wouldn't have skipped the download.
    """
    return f"https://github.com/{repo}/releases/download/latest-data/{asset_name}"


def _asset_url_if_newer(asset: dict, local_path: Path) -> str | None:
    """Return the asset API URL if remote is newer than local_path (or local is missing)."""
    from datetime import datetime, timezone
    remote_dt = datetime.fromisoformat(asset["updated_at"].replace("Z", "+00:00"))
    if local_path.exists():
        local_dt = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
        if remote_dt <= local_dt:
            return None
    # Use API URL (not browser_download_url) — required to auth private-repo assets.
    return asset["url"]


def _download_asset(url: str, dest: Path) -> bool:
    import httpx

    headers = {"Accept": "application/octet-stream"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=60)
        if resp.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            print(f"Downloaded {dest.name} from release ({len(resp.content)} bytes)")
            return True
    except Exception as e:
        print(f"Warning: could not download {dest.name} from release: {e}")
    return False


# Marker file we touch after every successful release-asset sync. The TTL
# gate reads this — NOT the DB's mtime — so a half-written stub or a local
# init that happens to land within the TTL window can't shadow the real
# release. A stale 60 KB ``data/olx_cars.db`` from a prior failed run
# was the live example: its recent mtime made ``_ensure_release_assets``
# treat it as fresh for two hours and silently bypass the GitHub API,
# leaving the dashboard pointing at an empty DB. (2026-05-03 fix.)
_RELEASE_CHECK_MARKER = PROJECT_ROOT / "data" / ".last_release_check"

# Production DB is ~90 MB. Anything under 1 MB is a stub (test init,
# half-written file from a partial download, sqlite empty schema). We
# use this to gate "is the local cache trustworthy" decisions — without
# it, a 60 KB stub fooled both ``_ensure_release_assets`` (treated as
# present, skipped CDN fallback) and the empty-state UI (loaded zero
# listings, looked like "no data yet" instead of "broken cache").
_DB_VALID_MIN_BYTES = 1_000_000


def _looks_like_real_db() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size >= _DB_VALID_MIN_BYTES


def _force_next_check():
    """Reset the check timer so the next _ensure_db() call hits GitHub API."""
    if _RELEASE_CHECK_MARKER.exists():
        try:
            _RELEASE_CHECK_MARKER.unlink()
        except OSError:
            pass


# Model + metrics live alongside the DB in the data release — shipped by CI
# (see .github/workflows/scrape.yml `train-model` step). The dashboard never
# trains locally; it just consumes what the pipeline produced.
_MODEL_PATH = PROJECT_ROOT / "data" / "price_model.joblib"
_METRICS_PATH = PROJECT_ROOT / "data" / "price_metrics.json"
_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "price_importance.json"

_RELEASE_ASSETS: tuple[tuple[str, Path], ...] = (
    ("olx_cars.db", DB_PATH),
    ("price_model.joblib", _MODEL_PATH),
    ("price_metrics.json", _METRICS_PATH),
    ("price_importance.json", _IMPORTANCE_PATH),
)


def _ensure_release_assets() -> bool:
    """Sync DB + model + metrics from the latest-data release (once per TTL).

    Returns True if the local DB exists after the attempt (model/metrics are
    best-effort — dashboard falls back to "no predictions" if they're missing).
    """
    import time

    # TTL gate is keyed on the marker we write after a successful API
    # sync — never on the DB's own mtime. A half-written stub or local
    # ``init_db`` would otherwise let the TTL silently shadow the real
    # release for two hours.
    if (
        _RELEASE_CHECK_MARKER.exists()
        and (time.time() - _RELEASE_CHECK_MARKER.stat().st_mtime) <= _CHECK_INTERVAL_SECONDS
    ):
        return DB_PATH.exists()

    repo = os.environ.get("GITHUB_REPOSITORY", "nikit34/olx-car-parser")
    if not repo:
        return _looks_like_real_db()

    assets = _list_release_assets(repo)
    if assets:
        for name, dest in _RELEASE_ASSETS:
            asset = assets.get(name)
            if not asset:
                continue
            url = _asset_url_if_newer(asset, dest)
            if url:
                _download_asset(url, dest)
        _RELEASE_CHECK_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _RELEASE_CHECK_MARKER.touch()
        if _looks_like_real_db():
            return True
        # API succeeded but our local DB is still suspicious (stub from
        # an earlier failed run, partial download, missing asset). Drop
        # through to the CDN path before giving up.

    # API failed / returned empty / left us with a stub DB. Fall through
    # to the public CDN URL — it has no API rate limit and works for
    # public release assets. We download unconditionally because we
    # lost the ability to compare remote.updated_at vs local.mtime;
    # on Streamlit Cloud cold-starts the local file is missing or stub
    # anyway, so this is the correct behaviour. Marker is NOT stamped
    # so the next call retries the API in case the rate-limit window
    # cleared (which would let us recover the mtime short-circuit).
    if not _looks_like_real_db():
        global _LAST_RELEASE_ERROR
        for name, dest in _RELEASE_ASSETS:
            _download_asset(_public_download_url(repo, name), dest)
        if _looks_like_real_db():
            # CDN download succeeded — clear any earlier API error so the
            # empty-state banner doesn't persist on a now-working dashboard.
            _LAST_RELEASE_ERROR = None
        elif not _LAST_RELEASE_ERROR:
            _LAST_RELEASE_ERROR = (
                "CDN fallback download did not produce a valid DB "
                f"(local size: {DB_PATH.stat().st_size if DB_PATH.exists() else 0} bytes)"
            )
    return _looks_like_real_db()


def _ensure_db() -> bool:
    """Back-compat alias — sync all release assets, report DB status."""
    return _ensure_release_assets()


def load_from_db() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Try loading real data from SQLite. Returns (listings_df, history_df) or None."""
    if not _ensure_db():
        return None

    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_listings_df, get_price_history_df

        init_db(str(DB_PATH))
        session = get_session()

        listings = get_listings_df(session)
        history = get_price_history_df(session)

        if listings.empty:
            return None

        return listings, history
    except Exception as e:
        print(f"Warning: could not load DB data: {e}")
        return None



def _sub_segment(fuel_type, engine_cc) -> str:
    """Market sub-segment from fuel type and engine displacement.

    E.g. "diesel_mid", "petrol_small", "electric".
    """
    import pandas as _pd
    fuel = "unk"
    if _pd.notna(fuel_type) and fuel_type:
        fl = str(fuel_type).lower()
        if "diesel" in fl:
            fuel = "diesel"
        elif "eléctrico" in fl or "electr" in fl:
            fuel = "electric"
        elif "híbrido" in fl or "hybrid" in fl:
            fuel = "hybrid"
        elif "gpl" in fl:
            fuel = "gpl"
        else:
            fuel = "petrol"
    if fuel == "electric":
        return "electric"
    if not _pd.notna(engine_cc) or not engine_cc or engine_cc <= 0:
        return fuel
    if engine_cc < 1400:
        return f"{fuel}_small"
    if engine_cc <= 2000:
        return f"{fuel}_mid"
    return f"{fuel}_large"


def _load_llm_extras(raw_extras) -> dict:
    """Best-effort parse of serialized LLM extras stored on a listing row."""
    if isinstance(raw_extras, dict):
        return raw_extras
    if not isinstance(raw_extras, str) or not raw_extras.strip():
        return {}
    try:
        parsed = json.loads(raw_extras)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalized_text_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if item not in (None, "")]


# Salvage / non-runner phrases that should hard-block a deal even when
# enrichment is stale (i.e. ``damage_severity`` was set under the old
# regex). Scans title + description: the 2026-05-02 audit cases JmUNP /
# JmutI / JmR3C all had the giveaway phrase only in the description, so a
# title-only scan would miss them. Title-or-description match is cheap on
# the active subset — ~3 MB of text on the 18 k-row release.
#
# The phrase set is the union of ``llm_enrichment._PARTS_ONLY_HARD_PATTERN``,
# ``_NON_RUNNER_HARD_PATTERN``, and ``_SEVERE_DAMAGE_PATTERN``. We re-list
# them rather than import to avoid an enrichment-module import on every
# dashboard refresh; staleness is acceptable because both modules are
# touched in lock-step.
_HARD_BLOCK_TEXT_PATTERN = re.compile(
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


def _blocking_deal_reason(listing: pd.Series) -> str | None:
    """Return a hard-stop reason for listings that should not be shown as deals.

    Five signals, evaluated in decreasing order of certainty:

    1. ``desc_mentions_accident`` (DB column).
    2. ``damage_severity >= 3`` (DB column, derived in
       :func:`src.parser.llm_enrichment._derive_damage_severity`) — covers
       parts-only / non-runner / salvage phrasings without an LLM round-trip.
    3. ``right_hand_drive`` (DB column) — the PT market doesn't accept RHD
       cars at any meaningful resale price, so they never qualify as deals.
    4. Regex scan over ``title`` for salvage phrasings — defense-in-depth
       for listings whose enrichment hasn't run yet (``damage_severity`` and
       ``mechanical_condition`` are both NULL on freshly scraped rows).
    5. ``mechanical_condition == "poor"`` and ``photo_damage_flagged`` (from
       ``llm_extras``). Photo decision uses :func:`is_listing_flagged` so
       the multi-photo agreement rule (issue #8) takes precedence over a
       single high-p frame for post-#2 listings, with a fall-back to the
       v2 max-rule for the ~6 271 pre-#2 rows we never backfilled.
    """
    from src.parser.damage_decision import is_listing_flagged

    desc_mentions_accident = listing.get("desc_mentions_accident")
    if pd.notna(desc_mentions_accident) and bool(desc_mentions_accident):
        return "description mentions accident"

    damage_severity = listing.get("damage_severity")
    if pd.notna(damage_severity):
        try:
            sev = int(damage_severity)
        except (TypeError, ValueError):
            sev = None
        if sev is not None and sev >= 3:
            return f"damage severity {sev} (salvage / parts-only)"

    right_hand_drive = listing.get("right_hand_drive")
    if pd.notna(right_hand_drive) and bool(right_hand_drive):
        return "right-hand drive (PT market mismatch)"

    title = listing.get("title") or ""
    description = listing.get("description") or ""
    haystack = f"{title} {description}" if isinstance(title, str) and isinstance(description, str) else ""
    if haystack.strip():
        m = _HARD_BLOCK_TEXT_PATTERN.search(haystack)
        if m:
            return f"salvage phrasing in text: '{m.group(0).lower()}'"

    extras = _load_llm_extras(listing.get("llm_extras"))
    if not extras:
        return None

    if str(extras.get("mechanical_condition") or "").strip().lower() == "poor":
        return "poor mechanical condition"

    if is_listing_flagged(extras):
        photo_p = extras.get("photo_damage_p")
        if isinstance(photo_p, (int, float)):
            return f"photo damage detected (p={photo_p:.2f})"
        return "photo damage detected"

    return None


def prepare_active_for_model(
    listings_df: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
    include_sold: bool = False,
) -> pd.DataFrame:
    """Listings enriched with generation, sub_segment, avg_days_to_sell.

    Shared by compute_signals (dashboard) and the `train-model` CLI so the
    model sees the exact same feature prep in both paths.

    ``include_sold=True`` keeps inactive listings whose
    ``deactivation_reason == "sold"`` in the result, alongside actives.
    Used by training to roughly 3× the row count (~6 k active vs ~12 k
    sold on the 2026-05-03 release); the price-model side of training
    applies a per-row target discount + sample-weight haircut to those
    rows so the unknown gap between last ask and actual sold price
    doesn't bias the model toward "what didn't sell".

    Dashboard scoring keeps the default (active-only) — we only need
    fair-price predictions for listings the user can actually buy.
    """
    from src.models.generations import get_generation
    from src.analytics.turnover import compute_turnover_stats

    if listings_df.empty:
        return listings_df.copy()

    if "is_active" in listings_df.columns:
        if include_sold:
            mask = listings_df["is_active"].astype(bool)
            if "deactivation_reason" in listings_df.columns:
                mask = mask | (
                    (~listings_df["is_active"].astype(bool))
                    & (listings_df["deactivation_reason"].astype(str) == "sold")
                )
            active = listings_df[mask].copy()
        else:
            active = listings_df[listings_df["is_active"]].copy()
    else:
        active = listings_df.copy()
    if "duplicate_of" in active.columns:
        active = active[active["duplicate_of"].isna()].copy()

    gen_keys = active[["brand", "model", "year"]].drop_duplicates()
    gen_map = {
        (b, m, y if pd.notna(y) else None): get_generation(b, m, y if pd.notna(y) else None)
        for b, m, y in gen_keys.itertuples(index=False, name=None)
    }
    active["generation"] = [
        gen_map.get((b, m, y if pd.notna(y) else None))
        for b, m, y in zip(active["brand"], active["model"], active["year"])
    ]

    def _sub_key(f, e):
        fk = f if (isinstance(f, str) and f) else None
        ek = float(e) if pd.notna(e) and e else None
        return (fk, ek)

    # Tolerate fixtures/tests that omit these optional columns — fall back
    # to None for the missing one so the original apply-based behavior holds.
    fuel_series = active["fuel_type"] if "fuel_type" in active.columns else pd.Series([None] * len(active), index=active.index)
    cc_series = active["engine_cc"] if "engine_cc" in active.columns else pd.Series([None] * len(active), index=active.index)

    sub_keys = pd.DataFrame({"fuel_type": fuel_series, "engine_cc": cc_series}).drop_duplicates()
    sub_map = {
        _sub_key(f, e): _sub_segment(f, e)
        for f, e in sub_keys.itertuples(index=False, name=None)
    }
    active["sub_segment"] = [
        sub_map.get(_sub_key(f, e))
        for f, e in zip(fuel_series, cc_series)
    ]

    if turnover is None:
        turnover = compute_turnover_stats(listings_df)
    liquidity_map: dict[tuple, float] = {}
    if not turnover.empty:
        for row in turnover.itertuples(index=False):
            days = getattr(row, "avg_days_to_sell", None)
            if pd.notna(days):
                gen = getattr(row, "generation", None)
                liquidity_map[(row.brand, row.model, gen if pd.notna(gen) else None)] = float(days)
    if liquidity_map:
        active["avg_days_to_sell"] = [
            liquidity_map.get((b, m, g if pd.notna(g) else None))
            for b, m, g in zip(active["brand"], active["model"], active["generation"])
        ]

    return active


def compute_signals(
    listings_df: pd.DataFrame,
    history_df: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find undervalued listings and rank by flip potential.

    Price model (gradient boosting) uses 20 features including LLM-extracted
    fields (accident, RHD, condition, etc.) to predict fair market value.

    Flip score = undervaluation % × opportunity multipliers:
    - Liquidity: how fast this model sells
    - Trend: market price direction
    - Motivated seller: long listing + price drops
    - Urgency: seller desperation signals
    - Warranty: easier resale with warranty
    - Velocity: fast-selling segments
    - Sample confidence: comparable listing count
    - Band confidence: how tight the model's own [low, high] band is — a
      30% discount on a row whose band spans ±30% is noise; a 15% discount
      on a row whose band spans ±5% is an actionable flip. Decile-CQR
      surfaces this honestly per price tier (cheap segment legitimately
      gets ±27% bands).
    """
    if listings_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    import numpy as np

    signals = []
    active = prepare_active_for_model(listings_df, turnover=turnover)

    # --- Generation-level stats ---
    priced_gen = active[(active["price_eur"] > 0) & active["generation"].notna()]
    gen_stats = (
        priced_gen
        .groupby(["brand", "model", "generation"])
        .agg(gen_median=("price_eur", "median"), gen_count=("price_eur", "count"),
             gen_year_median=("year", "median"), gen_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Model-level fallback stats (includes cars without generation) ---
    priced_all = active[active["price_eur"] > 0]
    model_stats = (
        priced_all
        .groupby(["brand", "model"])
        .agg(model_median=("price_eur", "median"), model_count=("price_eur", "count"),
             model_year_median=("year", "median"), model_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Sub-segment stats (fuel + engine size within generation) ---
    sub_stats = (
        priced_gen
        .groupby(["brand", "model", "generation", "sub_segment"])
        .agg(sub_median=("price_eur", "median"), sub_count=("price_eur", "count"),
             sub_year_median=("year", "median"), sub_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Liquidity map (keyed by (brand, model, generation|None)) ---
    from src.analytics.turnover import compute_turnover_stats
    if turnover is None:
        turnover = compute_turnover_stats(listings_df)
    liquidity_map: dict[tuple, float] = {}
    if not turnover.empty:
        for row in turnover.itertuples(index=False):
            days = getattr(row, "avg_days_to_sell", None)
            if pd.notna(days):
                gen = getattr(row, "generation", None)
                liquidity_map[(row.brand, row.model, gen if pd.notna(gen) else None)] = float(days)

    # --- Price trend from history (last 60 days) ---
    trend_map: dict[tuple, float] = {}
    if not history_df.empty and "date" in history_df.columns:
        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=60)
        recent = hist[hist["date"] >= cutoff]
        for (brand, model), group in recent.groupby(["brand", "model"]):
            sorted_g = group.sort_values("date")
            if len(sorted_g) < 2:
                continue
            old_med = sorted_g.iloc[0]["median_price_eur"]
            new_med = sorted_g.iloc[-1]["median_price_eur"]
            if old_med and old_med > 0:
                trend_map[(brand, model)] = round((new_med - old_med) / old_med * 100, 1)

    # --- Sale velocity: fraction of recently deactivated listings sold within 21 days ---
    velocity_map: dict[tuple, float] = {}
    inactive = listings_df[~listings_df["is_active"]].copy() if "is_active" in listings_df.columns else pd.DataFrame()
    if not inactive.empty and {"deactivated_at", "first_seen_at", "brand", "model"}.issubset(inactive.columns):
        sold = inactive[inactive["deactivated_at"].notna() & inactive["first_seen_at"].notna()].copy()
        if not sold.empty:
            sold["_lifespan"] = (
                pd.to_datetime(sold["deactivated_at"]) - pd.to_datetime(sold["first_seen_at"])
            ).dt.days
            sold = sold[sold["_lifespan"] > 0]
            group_keys = ["brand", "model", "generation"] if "generation" in sold.columns else ["brand", "model"]
            for keys, group in sold.groupby(group_keys, dropna=False):
                if len(group) < 3:
                    continue
                velocity_map[keys] = float((group["_lifespan"] <= 21).mean())

    # avg_days_to_sell already populated by prepare_active_for_model.

    # --- Gradient boosting price model (uses LLM fields + market data) ---
    from src.analytics.price_model import (
        predict_prices,
        compute_feature_completeness,
        load_importance,
        load_model,
    )

    feature_fill = compute_feature_completeness(active)

    # Model is trained in CI (see `train-model` CLI) and shipped in the data
    # release. Dashboard never trains inline — if the model is missing or
    # too old, we skip predictions instead of blocking the user for minutes.
    saved = load_model(max_age_hours=14 * 24)  # tolerate up to 2 weeks
    gb_models = None
    gb_cat_maps: dict = {}
    _gb_metrics: dict | None = None
    _gb_oof_preds: dict[str, tuple[float, float, float]] = {}
    _gb_calibrator = None
    if saved is not None:
        (
            gb_models, gb_cat_maps, _gb_metrics, _gb_oof_preds,
            _gb_calibrator,
        ) = saved
    else:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "No fresh price model available — skipping predictions. "
            "Run `python -m src.cli train-model` or wait for the next CI run."
        )

    gb_predictions: dict[str, float] = {}
    gb_fair_low: dict[str, float] = {}
    gb_fair_high: dict[str, float] = {}
    # Importance is computed once during training (CI) and shipped in the
    # data release — loading it here costs one JSON read instead of a
    # 690-predict permutation loop per signal recompute.
    importance_df = load_importance()
    if gb_models is not None:
        _conformal_q = _gb_metrics.get("conformal_q", 0.0) if _gb_metrics else 0.0
        _per_bucket_q = (
            _gb_metrics.get("conformal_q_per_bucket", {}) if _gb_metrics else {}
        )
        # Dynamic decile edges persisted at train time. Tuple format is
        # (low, high, label). joblib round-trip preserves Python tuples
        # but JSON serialisation in metrics_history may turn them into
        # lists — normalise either way.
        _bucket_edges_raw = (
            _gb_metrics.get("conformal_q_bucket_edges") if _gb_metrics else None
        )
        _bucket_edges = (
            [tuple(e) for e in _bucket_edges_raw] if _bucket_edges_raw else None
        )
        # OOF preds (built during CV training) override model.predict for any
        # listing the model was trained on, so the deal-scoring loop compares
        # asking price against an out-of-fold "fair price" rather than an
        # in-sample one. The calibrator only applies to non-OOF rows since
        # OOF preds already have isotonic calibration baked in.
        # conformal_q_per_bucket gives class-conditional widening — each
        # row's q depends on its predicted-price tier.
        price_df = predict_prices(
            gb_models, gb_cat_maps, active,
            conformal_q=_conformal_q,
            oof_preds=_gb_oof_preds,
            median_calibrator=_gb_calibrator,
            conformal_q_per_bucket=_per_bucket_q,
            conformal_q_bucket_edges=_bucket_edges,
        )
        if "olx_id" in active.columns:
            olx_ids = active["olx_id"].reindex(price_df.index).values
            preds = price_df["predicted_price"].values
            lows = price_df["fair_price_low"].values
            highs = price_df["fair_price_high"].values
            for oid, pred, lo, hi in zip(olx_ids, preds, lows, highs):
                if oid and pred > 0:
                    gb_predictions[oid] = float(pred)
                    gb_fair_low[oid] = float(lo)
                    gb_fair_high[oid] = float(hi)

    # --- Precompute stat lookups (dict by tuple) to avoid per-row boolean
    # indexing over sub_stats/gen_stats/model_stats in the scoring loop.
    sub_lookup: dict[tuple, tuple] = {
        (r.brand, r.model, r.generation, r.sub_segment):
            (r.sub_median, int(r.sub_count), r.sub_year_median, r.sub_mileage_median)
        for r in sub_stats.itertuples(index=False)
    }
    gen_lookup: dict[tuple, tuple] = {
        (r.brand, r.model, r.generation):
            (r.gen_median, int(r.gen_count), r.gen_year_median, r.gen_mileage_median)
        for r in gen_stats.itertuples(index=False)
    }
    model_lookup: dict[tuple, tuple] = {
        (r.brand, r.model):
            (r.model_median, int(r.model_count), r.model_year_median, r.model_mileage_median)
        for r in model_stats.itertuples(index=False)
    }

    # --- Score each listing ---
    for _, listing in active.iterrows():
        price = listing.get("price_eur")
        if pd.isna(price) or price <= 0:
            continue

        year = listing.get("year")
        mileage = listing.get("mileage_km")
        brand = listing["brand"]
        model = listing["model"]
        generation = listing.get("generation")

        # Resolve comparison group: sub-segment → generation → model
        median = None
        sample = 0
        group_year_median = None
        group_mileage_median = None
        sub_seg = listing.get("sub_segment")

        if generation and sub_seg:
            hit = sub_lookup.get((brand, model, generation, sub_seg))
            if hit is not None and hit[1] >= 5:
                median, sample, group_year_median, group_mileage_median = hit

        if median is None and generation:
            hit = gen_lookup.get((brand, model, generation))
            if hit is not None:
                median, sample, group_year_median, group_mileage_median = hit

        if median is None:
            hit = model_lookup.get((brand, model))
            if hit is None:
                continue
            median, sample, group_year_median, group_mileage_median = hit

        if not median or price >= median * 0.85:
            continue

        if _blocking_deal_reason(listing):
            continue

        # 1. Undervaluation: gradient boosting predicted price (now includes LLM features).
        # Quality-over-coverage: a listing only qualifies as a deal when the
        # GB model has produced a prediction *and* the asking price is
        # below it. The previous median-discount fallback let listings
        # through even when the model said "fairly priced", which the
        # 2026-05-02 audit traced to ~37 % of the false-positive top-30
        # (Mercedes CLA, BMW X2, Toyota C-HR all surfaced via that path).
        # Coverage drops on segments where the model never trains, but
        # those segments produced bad recommendations anyway.
        olx_id = listing.get("olx_id", "")
        predicted = gb_predictions.get(olx_id)
        if not predicted or predicted <= 0:
            continue
        undervaluation_pct = round((1 - price / predicted) * 100, 1)
        if undervaluation_pct <= 0:
            continue

        # Severity-2 listings (needs repair / accident history) aren't
        # blocked but get their flip_score basis discounted by an
        # estimated repair cost — a €4k flip thesis on a "junta queimada"
        # Punto evaporates once you book €1.5k of head-gasket work.
        # Non-severity-2 rows pass through unchanged (repair_cost == 0).
        sev_raw = listing.get("damage_severity")
        try:
            severity_int = int(sev_raw) if pd.notna(sev_raw) else None
        except (TypeError, ValueError):
            severity_int = None
        mech_condition = listing.get("mechanical_condition")
        repair_cost = _estimate_repair_cost(
            severity_int, mech_condition, float(predicted),
        )
        profit_after_repair = float(predicted) - float(price) - repair_cost
        adjusted_pct = round(profit_after_repair / float(predicted) * 100, 1)
        if adjusted_pct <= 0:
            continue
        discount_pct = round((1 - price / median) * 100, 1)

        # Price range from quantile regression
        fair_low = gb_fair_low.get(olx_id)
        fair_high = gb_fair_high.get(olx_id)
        fill_rate = float(feature_fill.loc[listing.name]) if listing.name in feature_fill.index else 0.0
        sample_conf = min(sample / 20, 1.0)
        completeness = round(0.6 * fill_rate + 0.4 * sample_conf, 3)

        # --- Opportunity multipliers (deal quality, not market value) ---

        # 2. Liquidity multiplier (30 days = 1.0 baseline).
        # Key matches how the map is built: (brand, model, generation|None),
        # with a 2-level fallback if the generation-specific stat is missing.
        gen_key = generation if pd.notna(generation) and generation else None
        days_to_sell = liquidity_map.get((brand, model, gen_key))
        if days_to_sell is None and gen_key is not None:
            days_to_sell = liquidity_map.get((brand, model, None))
        if days_to_sell and days_to_sell > 0:
            liquidity_mult = min(max(30 / days_to_sell, 0.5), 2.0)
        else:
            liquidity_mult = 1.0

        # 3. Trend multiplier (rising market = bonus)
        trend_pct = trend_map.get((brand, model), 0.0)
        trend_mult = min(max(1 + trend_pct / 100, 0.8), 1.2)

        # 4. Motivated seller — long listing + price drops = negotiation room
        days_listed = listing.get("days_listed") if "days_listed" in listing.index else None
        price_change = listing.get("price_change_eur") if "price_change_eur" in listing.index else None
        if pd.notna(days_listed) and days_listed > 30 and pd.notna(price_change) and price_change < 0:
            motivated_mult = min(1.0 + abs(float(price_change)) / float(price) * 0.5, 1.3)
        elif pd.notna(days_listed) and days_listed > 60:
            motivated_mult = 1.1
        else:
            motivated_mult = 1.0

        # 5. Urgency — desperate seller = negotiation opportunity
        urgency = listing.get("urgency")
        if pd.notna(urgency) and urgency == "high":
            urgency_mult = 1.3
        elif pd.notna(urgency) and urgency == "medium":
            urgency_mult = 1.1
        else:
            urgency_mult = 1.0

        # 6. Warranty — easier to resell with warranty
        has_warranty = listing.get("warranty")
        if pd.notna(has_warranty) and has_warranty:
            warranty_mult = 1.15
        else:
            warranty_mult = 1.0

        # 7. Sale velocity — fast-selling segments = better for flipping
        velocity = velocity_map.get((brand, model, listing.get("generation")))
        if velocity is not None:
            velocity_mult = min(max(0.7 + velocity * 0.8, 0.7), 1.5)
        else:
            velocity_mult = 1.0

        # 8. Sample confidence — more comparable listings = more reliable estimate
        if sample >= 10:
            confidence_mult = 1.2
        elif sample >= 7:
            confidence_mult = 1.1
        elif sample >= 5:
            confidence_mult = 1.0
        else:
            confidence_mult = 0.7

        # 9. Band-width gate — when the model's [low, high] band is wide
        # relative to its own median, the model is saying "I don't know".
        # Decile-CQR surfaces this honestly: <€3k bin gets ±27% bands
        # because salvage / commercial / orphan-brand cars cluster there.
        # We don't drop those rows (the user might still want to see them)
        # but we deprioritise them so tight-band high-confidence deals
        # dominate the top of the list. None when the bundle didn't ship
        # bands — gracefully no-op.
        if (
            fair_low is not None and fair_high is not None
            and predicted and predicted > 0
        ):
            band_pct = (fair_high - fair_low) / predicted
            if band_pct <= 0.15:
                band_confidence_mult = 1.15
            elif band_pct <= 0.25:
                band_confidence_mult = 1.0
            elif band_pct <= 0.40:
                band_confidence_mult = 0.7
            else:
                band_confidence_mult = 0.4
        else:
            band_pct = None
            band_confidence_mult = 1.0  # no band → don't penalise

        # Use repair-adjusted basis when severity 2 is in play; for
        # severity 0/1/None this is identical to ``undervaluation_pct``.
        flip_score = round(
            adjusted_pct * liquidity_mult * trend_mult * motivated_mult
            * urgency_mult * warranty_mult * velocity_mult * confidence_mult
            * band_confidence_mult, 1
        )

        # --- Build signal dict ---
        desc_mentions_accident = listing.get("desc_mentions_accident")
        desc_mentions_repair = listing.get("desc_mentions_repair")
        desc_mentions_num_owners = listing.get("desc_mentions_num_owners")
        desc_mentions_customs_cleared = listing.get("desc_mentions_customs_cleared")
        right_hand_drive = listing.get("right_hand_drive")
        engine_cc = listing.get("engine_cc")

        sig = {
            "olx_id": olx_id,
            "url": listing.get("url", ""),
            "brand": brand,
            "model": model,
            "year": year,
            "generation": "" if pd.isna(generation) else (generation or ""),
            "sub_segment": sub_seg or "",
            "price_eur": price,
            "predicted_price": round(predicted) if predicted and predicted > 0 else None,
            "fair_price_low": fair_low,
            "fair_price_high": fair_high,
            "data_completeness": completeness,
            "median_price_eur": round(median),
            "discount_pct": discount_pct,
            "undervaluation_pct": undervaluation_pct,
            "damage_severity": severity_int,
            "repair_cost_eur": round(repair_cost) if repair_cost > 0 else None,
            "est_profit_after_repair_eur": (
                round(profit_after_repair) if repair_cost > 0 else None
            ),
            "adjusted_undervaluation_pct": adjusted_pct,
            "engine_cc": int(engine_cc) if pd.notna(engine_cc) and engine_cc else None,
            "liquidity_mult": round(liquidity_mult, 2),
            "trend_mult": round(trend_mult, 2),
            "motivated_mult": round(motivated_mult, 2),
            "urgency_mult": round(urgency_mult, 2),
            "warranty_mult": round(warranty_mult, 2),
            "velocity_mult": round(velocity_mult, 2),
            "confidence_mult": round(confidence_mult, 2),
            "band_pct": round(band_pct * 100, 1) if band_pct is not None else None,
            "band_confidence_mult": round(band_confidence_mult, 2),
            "avg_days_to_sell": days_to_sell,
            "price_trend_pct": trend_pct,
            "flip_score": flip_score,
            "sample_size": sample,
            "city": listing.get("city", ""),
            "district": listing.get("district", ""),
            "mileage_km": mileage,
            "fuel_type": listing.get("fuel_type", ""),
            # LLM fields for display/warnings
            "desc_mentions_accident": bool(desc_mentions_accident) if pd.notna(desc_mentions_accident) else None,
            "desc_mentions_repair": bool(desc_mentions_repair) if pd.notna(desc_mentions_repair) else None,
            "desc_mentions_num_owners": (
                int(desc_mentions_num_owners)
                if pd.notna(desc_mentions_num_owners) and desc_mentions_num_owners
                else None
            ),
            "desc_mentions_customs_cleared": (
                bool(desc_mentions_customs_cleared)
                if pd.notna(desc_mentions_customs_cleared)
                else None
            ),
            "right_hand_drive": bool(right_hand_drive) if pd.notna(right_hand_drive) else None,
            "urgency": urgency if pd.notna(urgency) else None,
            "warranty": bool(has_warranty) if pd.notna(has_warranty) else None,
            "taxi_fleet_rental": bool(listing.get("taxi_fleet_rental")) if pd.notna(listing.get("taxi_fleet_rental")) else None,
            "first_owner_selling": bool(listing.get("first_owner_selling")) if pd.notna(listing.get("first_owner_selling")) else None,
        }
        for col in ("days_listed", "price_change_eur", "price_change_pct", "eur_per_km"):
            if col in listing.index:
                sig[col] = listing[col]
        signals.append(sig)

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("flip_score", ascending=False)

    # Per-listing GB predictions for the FULL active set, regardless of
    # whether the listing qualified as a deal. The deal-card scatter
    # band overlay needs predictions for every comparable listing in
    # the segment — not just the ones that scored above the
    # undervaluation threshold and made it into ``df``.
    predictions_df = pd.DataFrame({
        "olx_id": list(gb_predictions.keys()),
        "predicted_price": list(gb_predictions.values()),
        "fair_price_low": [gb_fair_low.get(o) for o in gb_predictions],
        "fair_price_high": [gb_fair_high.get(o) for o in gb_predictions],
    })
    return df, importance_df, predictions_df



def load_unmatched() -> pd.DataFrame:
    """Load unmatched listings from database."""
    if not _ensure_db():
        return pd.DataFrame()
    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_unmatched_df
        init_db(str(DB_PATH))
        session = get_session()
        return get_unmatched_df(session)
    except Exception as e:
        print(f"Warning: could not load unmatched: {e}")
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    """Load portfolio deals from database."""
    if not _ensure_db():
        return pd.DataFrame()
    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_portfolio_df
        init_db(str(DB_PATH))
        session = get_session()
        return get_portfolio_df(session)
    except Exception as e:
        print(f"Warning: could not load portfolio: {e}")
        return pd.DataFrame()



def load_all():
    """Load listings, history, signals, brand map, turnover, and portfolio."""
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats

    db_data = load_from_db()

    if db_data is not None:
        listings, history = db_data
        listings = enrich_listings(listings)
        # Use LLM-corrected mileage everywhere (sellers game filters with fake low values).
        # Sanity gate: existing rows include parse-errors like Honda Civic
        # JmuYR at 278_000_000 km (LLM mis-read "278 mil km"). Drop anything
        # above 1M km before the fillna so it never overrides the
        # structured ``mileage_km`` attribute. The same cap exists at
        # write-time in :func:`correct_listing_data`, but we keep the
        # read-time gate so legacy bad rows don't poison segment medians
        # while the DB hasn't been re-enriched.
        if "real_mileage_km" in listings.columns:
            real_km = listings["real_mileage_km"]
            plausible = (real_km > 0) & (real_km <= _SANITY_MAX_MILEAGE_KM)
            listings["mileage_km"] = real_km.where(plausible).fillna(listings["mileage_km"])
        turnover = compute_turnover_stats(listings)
        signals, importance, predictions = compute_signals(
            listings, history, turnover=turnover,
        )
    else:
        listings = pd.DataFrame()
        history = pd.DataFrame()
        signals = pd.DataFrame()
        importance = pd.DataFrame()
        predictions = pd.DataFrame()
        turnover = pd.DataFrame()

    portfolio = load_portfolio()

    brands_models: dict[str, list[str]] = {}
    if not listings.empty:
        pairs = listings[["brand", "model"]].drop_duplicates()
        for brand, grp in pairs.groupby("brand", sort=False):
            brands_models[brand] = grp["model"].tolist()

    unmatched = load_unmatched()

    return (
        listings, history, signals, brands_models, turnover,
        portfolio, unmatched, importance, predictions,
    )
