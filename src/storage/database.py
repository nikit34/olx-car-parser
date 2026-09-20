"""Database connection and initialization."""

import json
import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from sqlalchemy import text

from src.models.listing import Base
import src.models.photo  # noqa: F401 — register ListingPhoto with Base
import src.models.portfolio  # noqa: F401 — register PortfolioDeal with Base
import src.models.relist  # noqa: F401 — register RelistEvent with Base
import src.models.seller  # noqa: F401 — register Seller with Base

_engine = None
_Session = None


class DatabaseNotConfigured(RuntimeError):
    """Raised when no engine URL is available."""


def resolve_db_url(db_url: str | None = None) -> str:
    """The engine URL for this process.

    An explicit URL wins — that is how tests and one-off tooling point at
    their own database. Otherwise ``OLX_DB_URL`` (the scrape host's
    PostgreSQL) is required: there is no local-file fallback to drift onto.
    """
    if db_url:
        return db_url
    env_url = os.environ.get("OLX_DB_URL", "").strip()
    if not env_url:
        raise DatabaseNotConfigured(
            "OLX_DB_URL is not set. Point it at the scrape host's PostgreSQL, "
            "e.g. postgresql+psycopg://olx@localhost/olx_cars"
        )
    return env_url


def get_engine(db_url: str | None = None):
    global _engine
    if _engine is None:
        _engine = create_engine(resolve_db_url(db_url), echo=False,
                                pool_pre_ping=True)
    return _engine


def open_conn(db_url: str | None = None):
    """Statement-autocommitting connection for scripts that hand-write SQL."""
    return get_engine(db_url).connect().execution_options(
        isolation_level="AUTOCOMMIT"
    )


def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


def _get_table_columns(conn, table_name: str) -> set[str]:
    try:
        return {col["name"] for col in inspect(conn).get_columns(table_name)}
    except Exception:
        return set()


_MERGE_DIRECT = (
    "url", "brand", "model", "price_label", "vat_label",
    "vat_reclaimable", "body_type", "year", "registration_month", "mileage_km",
    "engine_cc", "horsepower", "power_kw", "fuel_type", "transmission",
    "co2_g_km", "seller_type", "country_code", "city", "zip_code",
    "photo_count", "image_url", "is_active", "deactivated_at", "is_damaged",
    "first_seen_at", "last_seen_at", "source", "external_id",
)

_MERGE_RENAMES = {"region": "district"}

_MERGE_TO_EXTRAS = ("model_group", "variant", "motor_type", "version",
                    "offer_type")


def _merge_import_listings(conn, batch: int = 2000) -> int:
    """Fold the old foreign-market table into ``listings``. Returns rows moved.

    The two tables existed because foreign rows were card-level and Portuguese
    rows were not. Once every reader opens the advert that difference is gone,
    and a second table is just a second place to forget to look.

    Identity is namespaced on the way in. ``olx_id`` is the platform's own ad
    id, which it has been since StandVirtual moved in beside OLX, but OLX and
    Kleinanzeigen both number their ads and nothing stops the same integer
    naming a car in Lisbon and one in Leipzig. Foreign rows therefore land as
    ``source:external_id`` while the Portuguese pair keeps its bare ids, so no
    existing row changes key and no future one can collide.

    Fields with no column of their own — the seller's own damage flag, the
    trim words each site spells differently — go to ``extras`` as JSON rather
    than being dropped, because a migration that loses data is not reversible
    by re-running it.

    Idempotent: rows already carrying their namespaced id are skipped, so an
    interrupted run resumes and a finished one is a no-op. The old table is
    dropped only when nothing is left to move.
    """
    import json as _json

    names = set(inspect(conn).get_table_names())
    if "import_listings" not in names:
        return 0

    have = {row[0] for row in conn.execute(
        text("SELECT olx_id FROM listings WHERE olx_id LIKE '%:%'"))}
    columns = _get_table_columns(conn, "import_listings")
    direct = [c for c in _MERGE_DIRECT if c in columns]
    packed = [c for c in _MERGE_TO_EXTRAS if c in columns]
    renamed = {k: v for k, v in _MERGE_RENAMES.items() if k in columns}
    select_cols = sorted(set(direct) | set(packed) | set(renamed)
                         | {"source", "external_id"}
                         | ({"price_eur"} if "price_eur" in columns else set()))

    rows = conn.execute(text(
        f"SELECT {', '.join(select_cols)} FROM import_listings")).mappings().all()
    moved = 0
    pending: list[dict] = []
    target = [c for c in direct if c != "source" and c != "external_id"]
    insert_cols = (["olx_id", "source", "external_id", "extras"]
                   + target + list(renamed.values()))
    prices: dict[str, float] = {}
    statement = text(
        f"INSERT INTO listings ({', '.join(insert_cols)}) "
        f"VALUES ({', '.join(':' + c for c in insert_cols)})")

    for row in rows:
        source = str(row.get("source") or "")
        external_id = str(row.get("external_id") or "")
        if not source or not external_id:
            continue
        olx_id = f"{source}:{external_id}"
        if olx_id in have:
            continue
        extras = {c: row[c] for c in packed if row.get(c) is not None}
        payload = {c: row.get(c) for c in target}
        payload.update({native: row.get(foreign) for foreign, native in renamed.items()})
        payload.update({
            "olx_id": olx_id,
            "source": source,
            "external_id": external_id,
            "extras": _json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else None,
        })
        if row.get("price_eur") is not None:
            prices[olx_id] = float(row["price_eur"])
        pending.append(payload)
        have.add(olx_id)
        if len(pending) >= batch:
            conn.execute(statement, pending)
            conn.commit()
            moved += len(pending)
            pending = []
    if pending:
        conn.execute(statement, pending)
        conn.commit()
        moved += len(pending)

    if prices:
        rows_by_id = conn.execute(text(
            "SELECT id, olx_id FROM listings WHERE olx_id LIKE \'%:%\'")).all()
        snapshots = [{"listing_id": lid, "price_eur": prices[oid]}
                     for lid, oid in rows_by_id if oid in prices]
        already = {r[0] for r in conn.execute(text(
            "SELECT DISTINCT listing_id FROM price_snapshots"))}
        snapshots = [s for s in snapshots if s["listing_id"] not in already]
        if snapshots:
            conn.execute(text(
                "INSERT INTO price_snapshots (listing_id, price_eur, scraped_at) "
                "VALUES (:listing_id, :price_eur, NOW())"), snapshots)
            conn.commit()

    conn.execute(text("DROP TABLE import_listings"))
    conn.commit()
    return moved


_SCHEMA_VERSION = 9  # bump when _migrate_columns or _dead_json_keys changes


def _read_schema_version(conn) -> int:
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS _schema_meta (version INTEGER NOT NULL)"
    ))
    # PostgreSQL does not autocommit DDL and ``_write_schema_version`` runs
    # on a fresh connection — without this commit the table it writes to
    # was never created.
    conn.commit()
    row = conn.execute(text("SELECT version FROM _schema_meta LIMIT 1")).fetchone()
    return int(row[0]) if row else 0


def _write_schema_version(conn, version: int):
    conn.execute(text("DELETE FROM _schema_meta"))
    conn.execute(text("INSERT INTO _schema_meta (version) VALUES (:v)"), {"v": version})


def init_db(db_url: str | None = None):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)

    # Schema migrations are idempotent but expensive on large DBs (full
    # SELECT over llm_extras + one ALTER per added column).  Gate on a
    # persisted schema_version so startup cost for scrape/enrich/dashboard
    # is a single integer read once the migration has been applied.
    with engine.connect() as conn:
        current = _read_schema_version(conn)
        if current >= _SCHEMA_VERSION:
            conn.commit()
            return engine

    # Migrate: add columns to existing listings table
    _migrate_columns = [
        ("country_code", "TEXT DEFAULT 'PT'"),
        ("external_id", "TEXT"),
        ("image_url", "TEXT"),
        ("price_label", "TEXT"),
        ("zip_code", "TEXT"),
        ("body_type", "TEXT"),
        ("power_kw", "INTEGER"),
        ("vat_label", "TEXT"),
        ("vat_reclaimable", "BOOLEAN"),
        ("is_damaged", "BOOLEAN"),
        ("extras", "TEXT"),
        ("generation", "TEXT"),
        ("desc_mentions_repair", "BOOLEAN"),
        ("desc_mentions_accident", "BOOLEAN"),
        ("real_mileage_km", "INTEGER"),
        ("desc_mentions_num_owners", "INTEGER"),
        ("desc_mentions_customs_cleared", "BOOLEAN"),
        ("llm_description_hash", "TEXT"),
        ("source", "TEXT DEFAULT 'olx'"),
        ("duplicate_of", "TEXT"),
        ("right_hand_drive", "BOOLEAN"),
        ("deactivated_at", "TIMESTAMP"),
        ("deactivation_reason", "TEXT"),
        ("urgency", "TEXT"),
        ("warranty", "BOOLEAN"),
        ("tuning_or_mods", "TEXT"),
        ("taxi_fleet_rental", "BOOLEAN"),
        ("first_owner_selling", "BOOLEAN"),
        ("mechanical_condition", "TEXT"),
        ("drive_type", "TEXT"),
        ("sub_model", "TEXT"),
        ("trim_level", "TEXT"),
        ("photo_count", "INTEGER"),
        ("description_length", "INTEGER"),
        # v2: LLM-inferred damage severity (0=pristine, 3=salvage/parts).
        # Backfilled by `python -m src.cli enrich` (the pending query
        # re-runs LLM on rows whose llm_extras has no damage_severity yet).
        ("damage_severity", "INTEGER"),
        # v3: seller-profile FK + per-listing trader-title claim +
        # seller profile URL pointer (used by the backfill job to resolve
        # seller_uuid from the profile page after scrape time).
        ("seller_uuid", "TEXT"),
        ("seller_displayed_as", "TEXT"),
        ("seller_profile_url", "TEXT"),
        # v4: structured origin (national/imported) from the OLX/SV "origin"
        # param — distinct from the dropped dead-LLM "imported" key.
        ("origin", "TEXT"),
        # v5: CO₂ emissions g/km (StandVirtual detail "co2_emissions") — ISV input.
        ("co2_g_km", "INTEGER"),
        # v6: actual scrape wall-clock, distinct from last_seen_at (= OLX
        # posted date). NULL on existing rows until the next scrape re-sees
        # them (~all within one deep run); lets us measure real scrape
        # freshness/coverage instead of misreading posted-date as staleness.
        ("last_scraped_at", "TIMESTAMP"),
    ]
    _migrate_unmatched_columns = [
        ("source", "TEXT DEFAULT 'olx'"),
    ]
    # Sellers-table additions. The table itself is created by ``create_all``
    # for fresh DBs; these ALTERs cover dev DBs that ran an earlier v3
    # migration before the bucketing/identity expansion landed. Production
    # has no v3 yet, so these are belt-and-suspenders.
    _migrate_seller_columns = [
        ("family_lifestyle_count", "INTEGER"),
        ("electronics_count", "INTEGER"),
        ("realestate_count", "INTEGER"),
        ("tools_industrial_count", "INTEGER"),
        ("pets_hobby_count", "INTEGER"),
        ("services_jobs_count", "INTEGER"),
        ("social_account_type", "TEXT"),
        ("has_user_photo", "BOOLEAN"),
        ("position_lat", "REAL"),
        ("position_lon", "REAL"),
    ]
    # Columns removed from ORM — drop from DB if present
    _drop_columns = [
        # old heuristic columns (replaced by desc_mentions_* equivalents)
        "needs_repair", "had_accident", "num_owners", "customs_cleared",
        "mileage_suspect", "estimated_repair_cost_eur",
        # never used in src/ (NB: "origin" was here as a dead column but is now a
        # live structured field — captured from the OLX/SV param, see v4 migrate)
        "registration_plate", "tires_condition",
        # removed LLM fields (zero price-model importance)
        "accident_details", "imported", "paint_condition", "service_history",
        "repair_details", "suspicious_signs", "issues",
        "reason_for_sale", "recent_maintenance",
    ]
    # Keys to strip from llm_extras JSON
    _dead_json_keys = {
        "accident_details", "imported", "paint_condition", "service_history",
        "repair_details", "suspicious_signs", "extras", "issues",
        "reason_for_sale", "recent_maintenance", "tires_condition",
        "accident_free", "legal_issues",
    }
    # Indexes that ADD COLUMN doesn't create automatically. ``create_all``
    # builds them on fresh DBs, but existing rows added via ALTER TABLE
    # need an explicit ``CREATE INDEX IF NOT EXISTS`` to match the ORM.
    _migrate_relist_columns = [
        ("photo_score", "REAL"),
    ]
    _migrate_indexes = [
        ("ix_listings_seller_uuid", "listings", "seller_uuid"),
        ("ix_listings_last_scraped_at", "listings", "last_scraped_at"),
        ("ix_listings_country_code", "listings", "country_code"),
        ("ix_listings_external_id", "listings", "external_id"),
    ]
    with engine.connect() as conn:
        existing_listing_columns = _get_table_columns(conn, "listings")
        for col_name, col_type in _migrate_columns:
            if col_name in existing_listing_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE listings ADD COLUMN {col_name} "
                    f"{col_type}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        try:
            conn.execute(text(
                "UPDATE listings SET country_code = 'PT' WHERE country_code IS NULL"))
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            _merge_import_listings(conn)
        except Exception:
            conn.rollback()
        for idx_name, table, column in _migrate_indexes:
            try:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column})"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        existing_unmatched_columns = _get_table_columns(conn, "unmatched_listings")
        for col_name, col_type in _migrate_unmatched_columns:
            if col_name in existing_unmatched_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE unmatched_listings ADD COLUMN {col_name} "
                    f"{col_type}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        existing_relist_columns = _get_table_columns(conn, "relist_events")
        for col_name, col_type in _migrate_relist_columns:
            if col_name in existing_relist_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE relist_events ADD COLUMN {col_name} "
                    f"{col_type}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        existing_seller_columns = _get_table_columns(conn, "sellers")
        for col_name, col_type in _migrate_seller_columns:
            if col_name in existing_seller_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE sellers ADD COLUMN {col_name} "
                    f"{col_type}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        # Drop dead columns
        listing_columns = _get_table_columns(conn, "listings")
        for col_name in _drop_columns:
            if col_name in listing_columns:
                try:
                    conn.execute(text(f"ALTER TABLE listings DROP COLUMN {col_name}"))
                    conn.commit()
                except Exception:
                    conn.rollback()
        # Clean llm_extras JSON: strip removed keys
        rows = conn.execute(
            text("SELECT id, llm_extras FROM listings WHERE llm_extras IS NOT NULL")
        ).fetchall()
        updated = 0
        for row_id, raw in rows:
            try:
                data = json.loads(raw)
                keys_present = set(data) & _dead_json_keys
                if not keys_present:
                    continue
                for k in keys_present:
                    del data[k]
                conn.execute(
                    text("UPDATE listings SET llm_extras = :extras WHERE id = :id"),
                    {"extras": json.dumps(data, ensure_ascii=False), "id": row_id},
                )
                updated += 1
            except (json.JSONDecodeError, TypeError):
                continue
        if updated:
            conn.commit()
        _write_schema_version(conn, _SCHEMA_VERSION)
        conn.commit()
    return engine
