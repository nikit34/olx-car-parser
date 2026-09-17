"""One search card, in this project's vocabulary, whatever site it came from.

Every card-level reader ends here. The fields are the columns of
``market_listings`` that a search result can actually fill, so a reader either
puts a value in a named field or it does not have one — there is no place to
stash a number that came from somewhere else.

``extras`` is the escape hatch, and a narrow one: what a single site says and
the others have no word for. It rides as JSON into one column, which is what
keeps a new classified from widening the table with fields that stay NULL for
every other source. A key that turns up on several sites and starts mattering
to a model earns a real column then; until it does, it costs nothing to carry.

Values are translated on the way in, not on the way out. A reader maps the
site's own vocabulary onto ``Diesel``/``Gasolina``/``Manual``/``Automática``
and ``Profissional``/``Particular`` before building a card, because the price
model, the deal builders and the pages are written against those words, and a
per-site dialect in the database is a per-site bug in every one of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


def merge_patch(row: dict, patch: dict, corrects: tuple[str, ...] = ()) -> dict:
    """Lay an advert's fields over a card without overwriting what it knew.

    The advert is richer but not uniformly better. Kleinanzeigen prints the
    exact mileage on the card and rounds it to a bracket on the advert, so a
    patch fills gaps by default, and only the keys a reader names in its own
    ``CORRECTS`` — the ones its card is known to get wrong — may replace a
    value. AutoScout24 names none, because its card and its advert come from
    one database and agree.

    ``extras`` is the exception and merges rather than replacing, so a card's
    note and an advert's note both survive. Returns the same dict it was given.
    """
    for key, value in (patch or {}).items():
        if value is None or value == "":
            continue
        if key == "extras":
            current = row.get("extras")
            if isinstance(current, str):
                try:
                    current = json.loads(current)
                except ValueError:
                    current = {}
            merged = dict(current or {})
            merged.update(value if isinstance(value, dict) else {})
            row["extras"] = merged
            continue
        if key in corrects or row.get(key) in (None, "", 0):
            row[key] = value
    return row


@dataclass
class MarketCard:
    source: str
    external_id: str
    url: str
    brand: str
    model: str
    country_code: str
    price_eur: float | None = None
    price_label: str | None = None
    vat_label: str | None = None
    vat_reclaimable: bool | None = None
    model_group: str | None = None
    variant: str | None = None
    motor_type: str | None = None
    version: str | None = None
    body_type: str | None = None
    offer_type: str | None = None
    year: int | None = None
    registration_month: str | None = None
    mileage_km: int | None = None
    engine_cc: int | None = None
    horsepower: int | None = None
    power_kw: int | None = None
    fuel_type: str | None = None
    transmission: str | None = None
    co2_g_km: int | None = None
    seller_type: str | None = None
    region: str | None = None
    city: str | None = None
    zip_code: str | None = None
    is_damaged: bool | None = None
    photo_count: int | None = None
    image_url: str | None = None
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        """The card as an upsert payload, with ``extras`` serialised.

        Empty extras become NULL rather than ``{}`` so that "this site told us
        nothing unusual" and "we never looked" read the same in the column,
        which is the truth: there is no third state worth a byte here.
        """
        row = asdict(self)
        extras = row.pop("extras", None)
        row["extras"] = json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else None
        return row
