"""Domain logic for LLM-extracted listing facts — no transport, no provider.

This module owns everything that turns a raw extraction into trustworthy DB
columns: the deterministic ``damage_severity`` derivation (regex over
title+description, no model call), sub_model validation against brand tech-tag
families, mileage sanity bounds, and the ``correct_listing_data`` /
``apply_corrections`` write path.

The transport used to live here too — a pool of local Ollama backends. That is
gone (retired 2026-07-24, removed from both machines 2026-08-23). The single
LLM entry point is now :mod:`src.parser.cloud_enrichment`, a Gemini →
OpenRouter cascade fed only by the top-K ranked deals from
:mod:`src.analytics.value_gate`. Keeping the domain rules here means both the
cloud path and the offline backfills apply exactly the same validation.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based damage_severity derivation (no LLM call)
# ---------------------------------------------------------------------------
# When a listing already has a populated `llm_extras` dict from a previous
# enrichment run but is missing the (newer) damage_severity field, we don't
# need a fresh LLM call to backfill it — the existing accident/repair/
# condition flags plus a keyword scan over title+description carry enough
# signal. This path costs nothing (no model call at all), and on the
# 30-listing oracle it matches the LLM's choice on damage_severity exactly.
# Schema: 0=pristine, 1=normal wear, 2=needs repair OR accident history,
#         3=salvage / parts-only / non-runner.

# A tighter parts-car pattern than _PARTS_CAR_PATTERN (above): the latter
# also matches plain "avariado"/"imobilizado" which on their own only mean
# "needs repair", not "selling for parts". This one only fires on phrasings
# that explicitly mark the listing as parts-only / scrap.
_PARTS_ONLY_HARD_PATTERN = re.compile(
    r"para\s+pe[çc]as|vender\s+as\s+pe[çc]as|venda\s+de\s+pe[çc]as|"
    r"vende[-\s]se\s+a?\s*pe[çc]as|"
    r"para\s+sucata|para\s+desmanchar|s[óo]\s+pe[çc]as|abate|"
    r"para\s+exporta(?:r|[çc][ãa]o).{0,40}pe[çc]as|"
    r"sem\s+matr[ií]cula|sem\s+documentos",
    re.IGNORECASE,
)

# Non-runner phrasings — car physically does not move under its own power.
# These are unconditionally severity 3: mechanical_condition might say
# "good" because the body is fine, but a tow-only car has no flip thesis.
# JmUNP (Peugeot 508 SW) hit top-30 at flip_score 69 because the original
# severe-damage path returned 2 unless mechanical_condition was also
# "poor", and the LLM had marked it "fair". Also covers "só reboque" /
# "apenas reboque" / "engine seized" — see 2026-05-02 audit notes.
_NON_RUNNER_HARD_PATTERN = re.compile(
    r"n[ãa]o\s+pega|n[ãa]o\s+anda|n[ãa]o\s+funciona|"
    r"(?:o\s+carro\s+)?n[ãa]o\s+liga|n[ãa]o\s+arranca|"
    r"n[ãa]o\s+(?:é\s+)?poss[ií]vel\s+test(?:ar|á-lo)|"
    r"non[\s-]runner|engine\s+seized|"
    r"(?:s[óo]|apenas)\s+(?:de\s+|com\s+)?reboque",
    re.IGNORECASE,
)

# Severe mechanical / structural damage that's not "selling for parts" and
# not (necessarily) a non-runner — the car is whole and may run but is
# seriously broken. Lands on severity 2 (or 3 if condition is also "poor").
# 2026-05-02 audit addition: ``junta queimada`` — Fiat Punto JmutI was
# selling at €350 with that in the description and surfaced at flip_score
# 54 under the old blocker.
_SEVERE_DAMAGE_PATTERN = re.compile(
    r"motor\s+(?:fundido|avariad[oa])|caixa\s+avariad[oa]|"
    r"transmiss[ãa]o\s+avariad[oa]|capotamento|"
    r"junta\s+(?:de\s+cabe[çc]a\s+)?queimada|"
    r"avaria\s+(?:no|do)\s+motor",
    re.IGNORECASE,
)

# Pristine-car signals — used to override the default-1 fallback when the
# extras dict from a previous LLM run didn't set mechanical_condition but
# the description is unmistakably positive. Captures "como novo", "estado
# impecável", "FULL EXTRAS" and the like — phrasings the oracle marks as
# damage_severity=0.
_PRISTINE_PATTERN = re.compile(
    r"como\s+novo|estado\s+impec[áa]vel|\bimpec[áa]vel\b|"
    r"excelente\s+estado|estado\s+excelente|"
    r"perfeito\s+estado|estado\s+perfeito|"
    r"irrepreens[íi]vel|estado\s+de\s+novo|"
    r"\bfull\s+extras\b",
    re.IGNORECASE,
)


def _derive_damage_severity(extras: dict, title: str, description: str) -> int:
    """Return damage_severity 0-3 from already-extracted extras + raw text.

    Used for the backfill path: a listing has llm_extras from a previous
    enrich run but lacks damage_severity (added in DB schema v2 / model v5).
    Re-running the LLM just to recover one integer per row is wasteful; the
    boolean flags + condition + keyword scan deliver the same signal.

    Decision order (first match wins):
      1. Explicit parts-only / no-plates phrasing → 3 (salvage)
      2. Non-runner phrasing (não pega, só reboque, …) → 3 unconditionally
      3. Severe mechanical text → 2 (and 3 if condition is also "poor")
      4. desc_mentions_accident OR desc_mentions_repair → 2
      5. mechanical_condition == "excellent" + no damage flags → 0
      6. mechanical_condition == "poor" → 2
      7. fall through → 1 (normal age-appropriate wear)
    """
    text = f"{title or ''} {description or ''}"
    if _PARTS_ONLY_HARD_PATTERN.search(text):
        return 3
    if _NON_RUNNER_HARD_PATTERN.search(text):
        return 3
    if _SEVERE_DAMAGE_PATTERN.search(text):
        return 3 if extras.get("mechanical_condition") == "poor" else 2

    # Existing flags carry the explicit accident/repair signal that the LLM
    # extracted on the previous pass. Inline the legacy aliases (had_accident,
    # needs_repair) so this helper can be called before _EXTRAS_KEY_ALIASES /
    # _get_extra are defined later in the module.
    accident = extras.get("desc_mentions_accident")
    if accident is None:
        accident = extras.get("had_accident")
    repair = extras.get("desc_mentions_repair")
    if repair is None:
        repair = extras.get("needs_repair")
    if accident or repair:
        return 2

    cond = extras.get("mechanical_condition")
    if cond == "poor":
        return 2
    if cond == "excellent":
        return 0
    # Positive-signal scan — oracle marks "como novo" / "FULL EXTRAS" /
    # "estado impecável" listings as 0 even when the previous LLM pass
    # didn't set mechanical_condition, so we look at the raw text.
    if _PRISTINE_PATTERN.search(text):
        return 0
    # Warranty mention without any damage flag — warranty implies a clean,
    # dealer-grade car most of the time. The structured `warranty` flag
    # from the previous LLM pass is more reliable than a raw "garantia"
    # token in the text (avoids "sem garantia" false positives).
    if extras.get("warranty") is True:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Schema documentation (kept as a Python list so consumers — eval scripts,
# annotation tools — share one source of truth for the field set).
# ---------------------------------------------------------------------------

# Mileage sanity bounds for the LLM-extracted ``mileage_in_description_km``.
# 1M km absolute cap covers every plausible odometer (the highest-mileage
# car ever recorded crossed ~5M km, but those don't show up on OLX). The
# relative gate fires when the LLM read picks up a malformed unit suffix —
# ``listing.mileage_km`` from the structured attribute is the trusted
# baseline, and any LLM read more than 10× larger *or* 10× smaller is a
# parse error. The downward direction is what caught 2026-05 cases like
# JltT9, where price-leaked-into-title ("BMW-520-f10 20129.000 €") made
# the LLM emit 9000 km against an attr of 355000.
_MILEAGE_SANITY_MAX_KM = 1_000_000
_MILEAGE_SANITY_RELATIVE_MAX = 10


_BRAND_FAMILY: dict[str, str] = {
    "Volkswagen": "VAG", "VW": "VAG", "Audi": "VAG",
    "Seat": "VAG", "SEAT": "VAG", "Skoda": "VAG", "Škoda": "VAG",
    "Peugeot": "PSA", "Citroen": "PSA", "Citroën": "PSA", "DS": "PSA",
    "Mercedes-Benz": "MB", "Mercedes": "MB",
    "Renault": "RNO", "Dacia": "RNO", "Nissan": "RNO",
    "Fiat": "FCA", "Alfa Romeo": "FCA", "Lancia": "FCA", "Jeep": "FCA",
    "Ford": "FORD",
    "Hyundai": "HK", "Kia": "HK",
    "Mazda": "MAZDA",
    "BMW": "BMW", "Mini": "BMW", "MINI": "BMW",
}

_TAG_FAMILY: dict[str, str] = {
    "tdi": "VAG", "tfsi": "VAG", "tsi": "VAG",
    "hdi": "PSA", "bluehdi": "PSA", "puretech": "PSA", "ehdi": "PSA",
    "cdi": "MB", "bluetec": "MB",
    "dci": "RNO", "tce": "RNO", "bluedci": "RNO", "digt": "RNO",
    "multijet": "FCA", "mjet": "FCA", "jtd": "FCA",
    "tdci": "FORD", "ecoboost": "FORD",
    "crdi": "HK",
    "skyactived": "MAZDA",
    # CDTI is GM/Opel — kept as its own family so non-Opel/non-GM brands
    # (e.g. Fiat with "1.3 CDTI") get rejected. Opel itself is omitted
    # from _BRAND_FAMILY because post-2017 it's PSA, so the validator
    # passes Opel CDTI/HDi/BlueHDi through without judgment.
    "cdti": "GM",
}

_TAG_RE = re.compile(
    r"\b(BlueHDi|BlueTec|Blue\s+dCi|Blue\s+HDi|PureTech|EcoBoost|"
    r"Multijet|M-?Jet|SkyActive-?D|DIG-?T|"
    r"TDI|TFSI|TSI|HDI|HDi|CDTI|CDI|dCi|DCI|TCe|TDCi|JTD|CRDi|e-HDi)\b",
    re.IGNORECASE,
)


def _validate_sub_model(brand: str, sub_model: str) -> str | None:
    """Drop sub_model whose tech tag belongs to the wrong brand family.

    Returns the original string when the brand is not in the family map,
    when no recognized tech tag is present, or when the tag's family
    matches the brand's. Returns None only on confirmed cross-family
    mismatch (e.g. "2.0 HDi" on an Audi).
    """
    if not brand or not sub_model:
        return sub_model
    brand_fam = _BRAND_FAMILY.get(brand)
    if not brand_fam:
        return sub_model
    m = _TAG_RE.search(sub_model)
    if not m:
        return sub_model
    tag_key = re.sub(r"[\s\-]", "", m.group(1).lower())
    tag_fam = _TAG_FAMILY.get(tag_key)
    if tag_fam and tag_fam != brand_fam:
        logger.info(
            "Dropped LLM sub_model %r for brand %s — tag family %s "
            "doesn't match brand family %s",
            sub_model, brand, tag_fam, brand_fam,
        )
        return None
    return sub_model


# ---------------------------------------------------------------------------
# Config / availability
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data correction — cross-check and fix attributes using LLM-extracted data
# ---------------------------------------------------------------------------

def correct_listing_data(listing) -> dict:
    """Cross-check listing attributes against LLM-extracted data and return corrections.

    Schema-v7-slim: LLM only returns sub_model, trim_level,
    mileage_in_description_km. damage_severity is derived deterministically
    via ``_derive_damage_severity`` (regex over title+description, with any
    legacy llm_extras flags taken into account when present).
    """
    extras = getattr(listing, "_llm_extras", None)
    if extras is None:
        return {}

    corrections = {}

    desc_km = extras.get("mileage_in_description_km")
    attr_km = listing.mileage_km

    if desc_km and isinstance(desc_km, (int, float)) and desc_km > 0:
        desc_km = int(desc_km)
        # Sanity gate: the LLM mis-parses "278 mil km" as 278000000 a few
        # times per 1k listings (Honda Civic JmuYR was the loudest case in
        # the 2026-05-02 audit at 278M km). Cap at 1M km — anything above
        # is a parse error, not a real odometer. Same gate fires when the
        # extracted value is >10× the structured attribute, which catches
        # mis-parsed unit-suffixes too.
        implausible_absolute = desc_km > _MILEAGE_SANITY_MAX_KM
        implausible_relative_high = (
            attr_km is not None
            and attr_km > 0
            and desc_km > attr_km * _MILEAGE_SANITY_RELATIVE_MAX
        )
        implausible_relative_low = (
            attr_km is not None
            and attr_km > 0
            and desc_km * _MILEAGE_SANITY_RELATIVE_MAX < attr_km
        )
        if implausible_absolute or implausible_relative_high or implausible_relative_low:
            logger.warning(
                "Implausible description mileage %d km (attr=%s) for %s — "
                "discarding LLM mileage",
                desc_km, attr_km, listing.url,
            )
            desc_km = None

    if desc_km and desc_km > 0:
        if attr_km and attr_km > 0:
            if desc_km > attr_km * 1.3 and (desc_km - attr_km) > 5000:
                corrections["real_mileage_km"] = desc_km
                logger.info(
                    "Mileage mismatch for %s: attribute=%d, description=%d → using description",
                    listing.url, attr_km, desc_km,
                )
            else:
                corrections["real_mileage_km"] = desc_km
        elif not attr_km or attr_km == 0:
            corrections["real_mileage_km"] = desc_km
    elif attr_km and attr_km > 0:
        corrections["real_mileage_km"] = attr_km

    sub_model = extras.get("sub_model")
    if sub_model and isinstance(sub_model, str) and sub_model.strip():
        validated = _validate_sub_model(
            getattr(listing, "brand", "") or "", sub_model.strip(),
        )
        if validated:
            corrections["sub_model"] = validated

    trim = extras.get("trim_level")
    if trim and isinstance(trim, str) and trim.strip():
        corrections["trim_level"] = trim.strip()

    title = getattr(listing, "title", "") or ""
    description = getattr(listing, "description", "") or ""
    corrections["damage_severity"] = _derive_damage_severity(
        extras, title, description,
    )

    return corrections


def apply_corrections(listings: list) -> int:
    """Apply data corrections to all listings that have LLM extras."""
    corrected = 0
    for listing in listings:
        corrections = correct_listing_data(listing)
        if not corrections:
            continue

        if not hasattr(listing, "_corrections"):
            listing._corrections = {}
        listing._corrections.update(corrections)
        corrected += 1

        if corrections.get("real_mileage_km") and corrections.get("real_mileage_km") != getattr(listing, "mileage_km", None):
            logger.info(
                "Corrected %s: real_mileage=%s, damage_severity=%s",
                listing.olx_id,
                corrections.get("real_mileage_km"),
                corrections.get("damage_severity"),
            )

    logger.info("Applied corrections to %d / %d listings", corrected, len(listings))
    return corrected


def merge_real_mileage(listings):
    """Overlay ``real_mileage_km`` (LLM description read) onto ``mileage_km``
    where the LLM value is sane.

    Mirrors the per-row gate in :func:`correct_listing_data`: a description
    read is trusted only when it's positive, within the absolute cap, and —
    if a structured attribute exists — within ``_SANITY_RELATIVE_MAX`` ratio
    of it. Without the relative gate, pre-2026-05-11 dirty-title rows
    (where the LLM read the price as mileage, e.g. JltT9's 9000 vs the
    real 355000) sneak through and the dashboard renders the price as km.

    Mutates and returns the DataFrame so callers can chain.
    """
    if "real_mileage_km" not in listings.columns:
        return listings
    real_km = listings["real_mileage_km"]
    attr_km = listings["mileage_km"]
    plausible_abs = (real_km > 0) & (real_km <= _MILEAGE_SANITY_MAX_KM)
    both_present = real_km.notna() & attr_km.notna() & (attr_km > 0)
    ratio_ok = (
        (real_km <= attr_km * _MILEAGE_SANITY_RELATIVE_MAX)
        & (real_km * _MILEAGE_SANITY_RELATIVE_MAX >= attr_km)
    )
    plausible = plausible_abs & (~both_present | ratio_ok)
    listings["mileage_km"] = real_km.where(plausible).fillna(attr_km)
    return listings
