"""OLX category-id snapshot for seller-profile feature engineering.

The OLX seller-profile JSON returns ad-count facets keyed by numeric
category id (e.g. ``{"id": 378, "count": 5}``). To roll those into stable
feature buckets without re-fetching the full ``categories.list`` map on
every parse, we hardcode the top-level ids and the immediate children of
the automotive root (id 362).

Snapshot date: 2026-05-05 (verified against four live profile pages and
one business shop). If OLX renumbers, the seller-profile parser will log
a warning when it encounters an unmapped facet id; counts still flow
into ``non_auto`` so the pipeline keeps working with degraded fidelity.
"""

from __future__ import annotations

# Top-level (parentId == 0) category ids on olx.pt.
TOP_LEVEL_IDS: dict[int, str] = {
    10: "Animais",
    11: "Tecnologia",
    12: "Desporto",
    13: "Móveis, Casa e Jardim",
    14: "Moda",
    16: "Imóveis",
    25: "Telemóveis, Tablets e Smartwatches",
    26: "Lazer",
    99: "Bebé e Criança",
    185: "Outras Vendas",
    190: "Emprego",
    191: "Serviços",
    362: "Carros, motos e barcos",
    4800: "Agricultura",
    4918: "Equipamentos e Ferramentas",
}

AUTO_ROOT_ID = 362

# Direct children of the automotive root (parentId == 362). Each carries
# a "feature bucket" label — the dimension we actually want at modelling
# time. ``parts`` groups everything that signals a dismantler/flipper:
# parts proper (377), parts-cars (5240), and write-offs/salvados (418).
AUTO_SUBTREE: dict[int, str] = {
    378: "cars",         # Carros (passenger cars)
    377: "parts",        # Peças e Acessórios
    5240: "parts",       # Carros para Peças
    418: "parts",        # Salvados (write-offs sold whole)
    416: "commercial",   # Comerciais - Camiões
    379: "motos",        # Motociclos - Scooters
    376: "boats",        # Barcos - Lanchas
    380: "other_auto",   # Outros veículos
    417: "other_auto",   # Autocaravanas - Reboques
}

CARS_CATEGORY_ID = 378  # immediate children of 378 are car-brand leaves.

# Non-automotive top-level groupings. Each bucket corresponds to a
# user "lifestyle" interpretation that's predictive at modelling time:
#
# * ``family_lifestyle`` — selling baby gear, used furniture, clothes →
#   reads as a real private seller offloading personal stuff. Listings
#   from these accounts tend to be honestly described.
# * ``electronics`` — phones+tablets+general tech. Heavy concentrations
#   indicate a tech reseller; the same person flipping a car is a flag.
# * ``realestate`` — Imóveis. Often appears next to a car listing when
#   the seller is relocating, which also implies "needs to sell now"
#   urgency that feeds into negotiation room.
# * ``tools_industrial`` — Equipamentos / Agricultura. Small contractors
#   and tradespeople; their cars are usually work vehicles with higher
#   wear than mileage suggests.
# * ``pets_hobby`` — animals, sport, leisure. Mostly noise but kept
#   isolated so feature ablation can decide.
# * ``services_jobs`` — Serviços, Emprego, Outras Vendas. Almost all
#   noise w.r.t. car-fraud signal, but worth pulling out separately so
#   it doesn't dilute the more informative buckets.
NON_AUTO_BUCKETS: dict[int, str] = {
    99: "family_lifestyle",      # Bebé e Criança
    13: "family_lifestyle",      # Móveis, Casa e Jardim
    14: "family_lifestyle",      # Moda
    11: "electronics",           # Tecnologia
    25: "electronics",           # Telemóveis, Tablets e Smartwatches
    16: "realestate",            # Imóveis
    4918: "tools_industrial",    # Equipamentos e Ferramentas
    4800: "tools_industrial",    # Agricultura
    10: "pets_hobby",            # Animais
    12: "pets_hobby",            # Desporto
    26: "pets_hobby",            # Lazer
    190: "services_jobs",        # Emprego
    191: "services_jobs",        # Serviços
    185: "services_jobs",        # Outras Vendas
}


def categorise_facets(
    facets: list[dict],
    categories_list: dict[str, dict] | None = None,
) -> dict[str, int]:
    """Roll a seller-profile facet list into stable feature buckets.

    Parameters
    ----------
    facets:
        Raw ``adsOffers.metadata.facets.category`` list from the seller-
        profile prerendered state. Each entry is ``{"id": int, "count":
        int, ...}``. The list mixes leaf categories (e.g. brand 741 BMW)
        with parent rollups (378 Carros, 362 Carros/motos/barcos) — OLX
        emits a row at every level of the tree that has a non-zero count.
    categories_list:
        Optional ``categories.list`` map (``{str(id): {"parentId": int,
        ...}}``) from the same payload. Required only for
        ``distinct_car_brands`` — that count needs the parent of each
        facet entry to determine which are direct children of 378.
        Passing ``None`` returns ``distinct_car_brands == 0``.

    Returns
    -------
    dict
        Counts per bucket: ``cars``, ``parts``, ``commercial``,
        ``motos``, ``boats``, ``other_auto``, ``non_auto``,
        ``distinct_car_brands``. ``cars`` etc. read directly from the
        rolled-up parent facet, NOT summed from brand leaves — OLX
        already gave us the rollup, and summing leaves would not match
        when a leaf goes missing from facets due to OLX's 150-entry cap.
    """
    out = {
        "cars": 0,
        "parts": 0,
        "commercial": 0,
        "motos": 0,
        "boats": 0,
        "other_auto": 0,
        "non_auto": 0,
        "distinct_car_brands": 0,
        # Non-auto sub-buckets — see NON_AUTO_BUCKETS for what each maps to.
        # ``non_auto`` stays as the sum across all of these so callers that
        # only want a single "is this a private user with other lives"
        # signal don't have to pull all six.
        "family_lifestyle": 0,
        "electronics": 0,
        "realestate": 0,
        "tools_industrial": 0,
        "pets_hobby": 0,
        "services_jobs": 0,
    }
    distinct_brands: set[int] = set()
    for f in facets or []:
        cid = f.get("id")
        count = f.get("count", 0)
        if cid is None or count is None:
            continue
        if cid in AUTO_SUBTREE:
            out[AUTO_SUBTREE[cid]] += count
        elif cid in NON_AUTO_BUCKETS:
            out[NON_AUTO_BUCKETS[cid]] += count
            out["non_auto"] += count
        elif cid in TOP_LEVEL_IDS and cid != AUTO_ROOT_ID:
            # Top-level not yet mapped to a bucket — bump the rollup
            # only, so the total stays accurate while we discover new
            # categories. Currently every top-level id is in
            # NON_AUTO_BUCKETS, so this branch is dead defence; a future
            # OLX schema change might reintroduce it.
            out["non_auto"] += count
        else:
            # leaf or sub-category — only count if it's a non-auto descendant.
            # If we have categories_list we can walk to the top; without it
            # we conservatively skip (rolled-up parent is in the same facet
            # list anyway, so we don't double-count).
            if categories_list is not None:
                top = _walk_to_top(cid, categories_list)
                if top is not None and top != AUTO_ROOT_ID:
                    # don't double-count: we already added the top-level rollup
                    pass
                # detect direct children of the cars node (378) — distinct brands
                parent = (categories_list.get(str(cid)) or {}).get("parentId")
                if parent == CARS_CATEGORY_ID:
                    distinct_brands.add(cid)
    out["distinct_car_brands"] = len(distinct_brands)
    return out


def _walk_to_top(cid: int, categories_list: dict[str, dict]) -> int | None:
    """Walk parentId chain in *categories_list* until parentId is 0/None."""
    cur = cid
    seen: set[int] = set()
    while cur and cur not in seen:
        seen.add(cur)
        node = categories_list.get(str(cur))
        if not node:
            return None
        parent = node.get("parentId")
        if parent in (0, None):
            return cur
        cur = parent
    return None
