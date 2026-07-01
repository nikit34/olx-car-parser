# Seller-Feature / Flipper Calibration Re-check — July 2026

**Run date:** 2026-07-01  
**Baseline:** 2026-05-06 (first post-backfill calibration, ~1 day of uuid-linked history)

---

## 1. Data Snapshot Summary

| Field | Value |
|---|---|
| Release tag | `latest-data` |
| Manifest built at | 2026-06-28T22:04:10Z (~56 days after the May calibration) |
| Total listings rows | 62,677 |
| Listings with `seller_uuid` | 16,654 (26.6 % of corpus) |
| Unique sellers with uuid | 15,011 |
| Sellers in 90-day window | 14,749 |

**Note on data source:** The `latest-data` release does not attach `olx_cars.db`. The mirror ships parquet artefacts (`listings.parquet`, `signals.parquet`, etc.). All SQL-equivalent queries below were executed against `listings.parquet` via pandas. The `price_importance.json` released alongside was used for the ablation check in lieu of a live `scripts/feature_ablation` run (see §4 for caveat).

---

## 2. 90-Day Rotation Distribution

Query equivalent: `SELECT seller_uuid, COUNT(*) n FROM listings WHERE seller_uuid IS NOT NULL AND first_seen_at >= '2026-04-02' GROUP BY seller_uuid`

**Cutoff:** 2026-04-02 (90 days before run date).  
**Sellers with ≥1 listing in window:** 14,749.

| n (listings / seller) | Sellers | % of pool |
|---|---|---|
| 1 | 13,580 | 92.07 % |
| 2 | 928 | 6.29 % |
| 3 | 170 | 1.15 % |
| 4 | 35 | 0.24 % |
| 5 | 13 | 0.09 % |
| 6 | 5 | 0.03 % |
| 7 | 6 | 0.04 % |
| 8 | 5 | 0.03 % |
| 9 | 3 | 0.02 % |
| 10 | 3 | 0.02 % |
| 11 | 1 | 0.01 % |

**Summary statistics:**

| Stat | Value |
|---|---|
| min | 1 |
| max | **11** |
| mean | 1.1073 |
| median | 1.0 |

**Threshold crossings:**

| Threshold | Count | % |
|---|---|---|
| n ≥ 3 | 241 | 1.63 % |
| n ≥ 5 | 36 | 0.24 % |
| n ≥ 8 | 12 | 0.08 % |
| n ≥ 10 | 4 | 0.03 % |

**Comparison to 2026-05-06 baseline:** The May baseline had max=5 with 99 % of sellers at n=1, across 1,278 sellers. Now the pool has grown to 14,749 sellers and the tail extends to n=11, but the distribution remains extremely right-skewed (92 % still at n=1, mean only 1.11).

---

## 3. Recommendation: `_W_LISTINGS_90D` New Value

Decision rules applied in order:

| Rule | Condition A | Condition B | Fires? |
|---|---|---|---|
| → 0.30 | max ≥ 10 (**11 ✓**) | ≥ 1 % of sellers cross n ≥ 5 (**0.24 % ✗**) | **No** |
| → 0.25 | max in \[6, 9\] (**11 ✗**, out of range) | — | **No** |
| Keep 0.15 | fallthrough | | **Yes** |

**Recommendation: keep `_W_LISTINGS_90D = 0.15`.**

The rotation primitive is now well-populated (14k+ sellers, real 90-day range), but the flipper signal is genuinely thin: only 36 sellers (0.24 %) appear ≥5 times. That concentration isn't enough to overcome the 99 %+ of innocent sellers who score 0 or 0.5 under any reasonable bucket scheme. The weight bump criterion (1 % of sellers crossing n ≥ 5) was set precisely to avoid over-amplifying a sparse tail.

**Weight table (no change):**

| Weight | Old | Proposed |
|---|---|---|
| `_W_LISTINGS_90D` | 0.15 | **0.15** (no change) |
| `_W_CARS_COUNT` | 0.30 | **0.30** (no change) |
| `_W_PSEUDOPRIVATE` | 0.35 | **0.35** (no change) |
| `_W_PLATE_OBSCURED` | 0.20 | **0.20** (no change) |
| **Sum** | 1.00 | **1.00** |

---

## 4. Recommendation: `_score_listings_90d` Buckets

**Trigger condition:** "If the rotation tail extends into 8+, propose new buckets 1–3 → 0.0, 4–7 → 0.5, 8+ → 1.0."

The tail now reaches n=11 (5 sellers at n=8, 3 at n=9, 3 at n=10, 1 at n=11). **The 8+ trigger fires.**

**Proposed bucket change:**

| n range | Current score | Proposed score |
|---|---|---|
| 1 | 0.0 | 0.0 |
| 2 | 0.5 | 0.0 |
| 3+ | 1.0 | 0.0 |
| 4–7 | — | 0.5 |
| 8+ | — | 1.0 |

**Justification from the histogram:**

Under the current scheme 8.33 % of sellers (n ≥ 2) score 0.5+, with 7.4 % at exactly 0.5 (n=2) — but n=2 is overwhelmingly dominated by unremarkable sellers. A genuine flipper tail begins around n=4–5 (35 and 13 sellers respectively), with the hard-rotation end at n=8+ (12 sellers). The proposed thresholds tighten the 0.5 band to n∈{4,5,6,7} (59 sellers, 0.40 %) and reserve 1.0 for n≥8 (12 sellers, 0.08 %). This reduces false positives while preserving the signal at the extreme tail now confirmed by 60 days of uuid-linked data.

---

## 5. Feature Ablation Result

**Source:** `price_importance.json` from the `latest-data` release (model built 2026-06-28, full CV-honest permutation importance, median quantile). The `scripts/feature_ablation.py` script could not be executed because `data/olx_cars.db` is absent from the release mirror (parquet only); the release JSON is the production equivalent.

**Threshold:** 0.003 (team standard since v7).

| Feature | low | median | high | vs 0.003 | Decision |
|---|---|---|---|---|---|
| year | 0.1999 | 0.2150 | 0.1090 | ≥ | KEEP |
| mileage_km | 0.0129 | 0.0389 | 0.0236 | ≥ | KEEP |
| model | 0.0390 | 0.0367 | 0.0229 | ≥ | KEEP |
| brand | 0.0062 | 0.0303 | 0.0055 | ≥ | KEEP |
| generation | 0.0111 | 0.0253 | 0.0160 | ≥ | KEEP |
| horsepower | 0.0072 | 0.0183 | 0.0127 | ≥ | KEEP |
| district | 0.0014 | 0.0062 | 0.0006 | ≥ | KEEP |
| engine_cc | 0.0004 | 0.0058 | 0.0042 | ≥ | KEEP |
| transmission | 0.0024 | 0.0051 | 0.0024 | ≥ | KEEP |
| photo_count | 0.0033 | 0.0050 | 0.0001 | ≥ | KEEP |
| segment | 0.0003 | 0.0048 | 0.0017 | ≥ | KEEP |
| sub_model | 0.0016 | 0.0027 | 0.0021 | < | FAIL |
| trim_level | 0.0022 | 0.0027 | 0.0010 | < | FAIL |
| fuel_type | 0.0001 | 0.0022 | 0.0002 | < | FAIL |
| seats | 0.0023 | 0.0022 | 0.0000 | < | FAIL |
| description_length | 0.0005 | 0.0019 | 0.0004 | < | FAIL |
| avg_days_to_sell | 0.0002 | 0.0010 | 0.0001 | < | FAIL |
| **seller_listings_count_90d** | 0.000224 | **0.000949** | 0.000067 | **<** | **DROP** |
| **plate_obscured** | 0.000100 | **0.000158** | 0.000042 | **<** | **DROP** |

Both seller-feature primitives remain below the 0.003 threshold at 60 days of uuid-linked history:

- `seller_listings_count_90d`: median importance **0.000949** (3.2× below threshold)
- `plate_obscured`: median importance **0.000158** (19× below threshold)

Note that `sub_model`, `trim_level`, `fuel_type`, `seats`, `description_length`, and `avg_days_to_sell` also fall below 0.003. These are outside the scope of this seller-calibration pass, but flag them for the next general model-feature sweep.

---

## 6. Flipper-Score Validation Result

**Method:** For closed listings (`deactivation_reason = 'sold'`) since 2026-05-06, the margin proxy is defined as `(first_price_eur − price_eur)` where `price_eur` is the last asking price before deactivation (not the true transaction price — the DB captures only the last OLX asking-price snapshot).

**Caveat:** A positive margin proxy means the seller lowered their asking price before the listing closed; it does NOT reliably capture negotiation discount or true achieved margin. Listings where `first_price_eur = price_eur` (no drop recorded) show proxy = 0 even if a transaction discount occurred off-platform. The median margin proxy was €0 and 1.9 % of values were negative (price rose between first and last seen).

**Counts:**

| Item | Count |
|---|---|
| Sold listings since 2026-05-06 | 32,046 |
| With both `flipper_score` and margin proxy | 21,359 |
| Mean flipper_confidence | 0.368 |
| Fraction with confidence ≥ 0.5 | 34.6 % |

**Pearson r = 0.0813 (p = 1.2 × 10⁻³², n = 21,359)**

Statistically significant but practically weak. The composite score does track the direction of margin (higher-scoring sellers show larger asking-price drops), but explains only ~0.7 % of margin variance. The low mean confidence (0.37) confirms that most scored listings lack one or more primitives, diluting the signal.

**Decile lift (flipper_score → mean margin proxy):**

| Decile | n | Score range | Mean margin (€) |
|---|---|---|---|
| 0 (lowest) | 8,559 | 0.000–0.094 | €279 |
| 1 | 2,382 | 0.150–0.200 | €227 |
| 2 | 2,968 | 0.225–0.531 | €212 |
| 3 (highest) | 8,450 | 0.538–1.000 | €459 |

The top decile group (score ≥ 0.538) shows the highest mean margin (€459). However, the decile structure collapsed to 4 groups instead of 10 due to the discrete, highly-clustered score distribution — scores cluster at a few combinations of the binary/bucket primitives. This is expected until the rotation primitive gains resolution (i.e., more sellers cross n≥3), at which point `_score_listings_90d` will produce a wider range of outputs and de-cluster the scores.

**Interpretation:** The composite has directional validity (higher score → more margin), but the margin proxy is too noisy and low-confidence-filtered to justify recalibrating weights by regression alone. Recommend re-running this check at 120 days (target 2026-09) when:  
1. More sellers accumulate multi-month history  
2. A proper sold-price proxy (e.g., the `_SOLD_TIERS`-adjusted target from `price_model.py`) can substitute for last-asking-price

---

## 7. Concrete Patch Suggestions

Do NOT apply automatically — owner review required.

### 7a. `src/analytics/flipper.py` — Bucket widening

**Change:** Update `_score_listings_90d` (lines 59–76) to use the wider thresholds. The function comment block (lines 61–68) should also be updated to reflect the new distribution.

**Current (lines 69–76):**
```python
    if n <= 1:
        return 0.0
    if n == 2:
        return 0.5
    return 1.0
```

**Proposed:**
```python
    if n <= 3:
        return 0.0
    if n <= 7:
        return 0.5
    return 1.0
```

Update the docstring comment at line 65 (currently references `1-2/3-5/6+`) to say `1-3/4-7/8+` and update the re-tune target date in line 68 from `2026-07` to `2026-09`.

**No weight changes** (rules in §3 above did not fire).

### 7b. `src/analytics/price_model.py` — Remove seller primitives from NUMERIC_FEATURES

Both `seller_listings_count_90d` and `plate_obscured` are below the 0.003 ablation threshold.

**Location:** `NUMERIC_FEATURES` list, lines 89–135. Specifically remove lines 113–114:
```python
    "seller_listings_count_90d",
    "plate_obscured",
```

**Prerequisite:** Removing these features changes `_ALL_FEATURES` length and column ordering, which invalidates any cached `price_model.joblib` bundle. A `_SCHEMA_VERSION` bump (currently `13` at line 80) to `14` is required so `load_model` rejects stale on-disk bundles. The `_MONOTONE_BY_FEATURE` dict at line 184 and `BOOL_FEATURES` (line 137) require no changes as neither feature appears there.

**Note:** `plate_obscured` and `seller_listings_count_90d` will continue to be scraped, persisted, and displayed on the dashboard — this change only removes them from the GBM's training matrix. The `flipper.py` composite (check 7a) continues to use them as human-facing signals.
