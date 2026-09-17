"""``get_country_listings_df``: a foreign corpus in the Portuguese frame's shape.

The price model, the model pages and the deal builders were written against
``get_listings_df``. A country corpus is useful only if it comes back with the
same columns under the same names, so most of this file compares the two frames
rather than inspecting the foreign one on its own. The rest pins what the
mapping decides: the source-prefixed id, the region standing in for the
district, the "expired" reason on retired rows, and the empty frame that still
knows its columns.
"""

from __future__ import annotations

import pandas as pd

from src.storage.repository import (
    deactivate_import_missing,
    get_country_listings_df,
    get_listings_df,
    upsert_import_listings,
    upsert_listing,
)

EXTRA_COLUMNS = {
    "country", "external_id", "image_url", "price_label", "co2_g_km",
    "registration_month", "vat_reclaimable", "is_damaged", "origin",
}


def _rows(source="as24_de", n=2, **over):
    from src.countries import code_for_source
    prefix = source.split("_")[-1]
    rows = []
    for i in range(n):
        row = {
            "source": source, "external_id": f"{prefix}-{i}",
            "url": f"https://www.autoscout24.{prefix}/angebote/golf-{i}",
            "brand": "Volkswagen", "model": "Golf", "version": "2.0 TDI DSG",
            "year": 2018, "price_eur": 18000 + i, "mileage_km": 90000 + i,
            "fuel_type": "Diesel", "horsepower": 150, "transmission": "Automática",
            "motor_type": "2.0 TDI", "body_type": "Kombi", "region": "Bayern",
            "city": "München", "seller_type": "Profissional", "photo_count": 15,
            "image_url": "https://prod.pictures.autoscout24.net/g/720x540.webp",
            "price_label": "Guter Preis", "co2_g_km": 120,
            "registration_month": "05/2018", "vat_reclaimable": True,
            "is_damaged": False, "country_code": code_for_source(source),
        }
        row.update(over)
        rows.append(row)
    return rows


class TestCountryListingsFrame:

    def test_it_has_every_column_the_portuguese_frame_has(self, db_session,
                                                           sample_listing_data):
        upsert_listing(db_session, sample_listing_data)
        db_session.commit()
        upsert_import_listings(db_session, _rows())
        pt = get_listings_df(db_session)
        de = get_country_listings_df(db_session, "DE")
        assert not pt.empty
        assert set(de.columns) >= set(pt.columns)
        assert EXTRA_COLUMNS <= set(de.columns)
        assert len(de) == 2

    def test_ids_are_prefixed_with_the_source(self, db_session):
        upsert_import_listings(db_session, _rows())
        de = get_country_listings_df(db_session, "DE")
        assert set(de["olx_id"]) == {"as24_de:de-0", "as24_de:de-1"}
        assert set(de["external_id"]) == {"de-0", "de-1"}

    def test_the_card_lands_in_the_portuguese_columns(self, db_session):
        upsert_import_listings(db_session, _rows(n=1))
        row = get_country_listings_df(db_session, "DE").iloc[0]
        assert row["title"] == "Volkswagen Golf 2.0 TDI DSG"
        assert row["district"] == "Bayern"
        assert row["segment"] == "Kombi"
        assert row["sub_model"] == "2.0 TDI"
        assert row["country"] == "DE"
        assert row["source"] == "as24_de"
        assert row["price_eur"] == 18000 and row["first_price_eur"] == 18000
        assert row["num_price_drops"] == 0 and row["max_drop_pct"] == 0.0
        assert pd.isna(row["price_drop_velocity"]) and pd.isna(row["days_since_last_drop"])
        assert row["last_scraped_at"] == row["last_seen_at"]
        assert row["seller_type"] == "Profissional"
        assert row["co2_g_km"] == 120 and row["registration_month"] == "05/2018"
        assert bool(row["vat_reclaimable"]) is True and bool(row["is_damaged"]) is False
        for absent in ("generation", "description", "llm_extras", "duplicate_of",
                       "seller_uuid", "origin", "seller_is_business"):
            assert pd.isna(row[absent]), absent

    def test_a_title_without_a_version_is_brand_and_model(self, db_session):
        upsert_import_listings(db_session, _rows(n=1, version=None))
        assert get_country_listings_df(db_session, "DE")["title"].iloc[0] == "Volkswagen Golf"

    def test_only_that_country_s_rows_come_back(self, db_session, sample_listing_data):
        upsert_listing(db_session, sample_listing_data)
        db_session.commit()
        upsert_import_listings(db_session, _rows("as24_de") + _rows("as24_fr", n=3)
                               + _rows("autoscout24", n=1))
        de = get_country_listings_df(db_session, "DE")
        fr = get_country_listings_df(db_session, "FR")
        assert set(de["source"]) == {"as24_de", "autoscout24"} and len(de) == 3
        assert set(fr["source"]) == {"as24_fr"} and len(fr) == 3
        assert get_country_listings_df(db_session, "IT").empty
        assert set(get_listings_df(db_session)["olx_id"]) == {"test-001"}

    def test_the_code_is_case_insensitive(self, db_session):
        upsert_import_listings(db_session, _rows())
        assert len(get_country_listings_df(db_session, "de")) == 2
        assert get_country_listings_df(db_session, "de")["country"].iloc[0] == "DE"

    def test_a_retired_row_reads_as_expired(self, db_session):
        upsert_import_listings(db_session, _rows())
        deactivate_import_missing(db_session, "as24_de", [("Volkswagen", "Golf", 2018)],
                                  {"de-0"})
        de = get_country_listings_df(db_session, "DE").set_index("external_id")
        assert bool(de.loc["de-1", "is_active"]) is False
        assert de.loc["de-1", "deactivation_reason"] == "expired"
        assert not pd.isna(de.loc["de-1", "deactivated_at"])
        assert bool(de.loc["de-0", "is_active"]) is True
        assert pd.isna(de.loc["de-0", "deactivation_reason"])

    def test_the_empty_frame_still_has_the_columns(self, db_session):
        empty = get_country_listings_df(db_session, "IT")
        assert empty.empty
        upsert_import_listings(db_session, _rows())
        full = get_country_listings_df(db_session, "DE")
        assert list(empty.columns) == list(full.columns)
        for col in ("olx_id", "district", "segment", "seller_is_business", "country"):
            assert col in empty.columns
