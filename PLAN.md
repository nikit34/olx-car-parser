# Implementation Plan

## Phase 1: Core Parser (MVP)
> Goal: scrape OLX car listings and save to SQLite

- [ ] **1.1** Set up Pydantic models for Listing and PriceSnapshot
- [ ] **1.2** Set up SQLAlchemy models + SQLite database initialization
- [ ] **1.3** Implement OLX search page scraper (pagination, listing URLs extraction)
- [ ] **1.4** Implement individual listing page parser (extract all car attributes)
- [ ] **1.5** Anti-blocking: request delays, User-Agent rotation, error handling
- [ ] **1.6** CLI command: `olx-parser scrape` - run a full scrape cycle
- [ ] **1.7** Test on real OLX pages, handle edge cases

## Phase 2: Recurring Scraping
> Goal: automate data collection on schedule

- [ ] **2.1** APScheduler integration - scrape every N hours
- [ ] **2.2** Deduplication logic: upsert listings by olx_id, append price snapshots
- [ ] **2.3** Track listing lifecycle (first_seen, last_seen, is_active)
- [ ] **2.4** CLI command: `olx-parser run` - start scheduler daemon
- [ ] **2.5** Logging and error notifications

## Phase 3: Price Analytics
> Goal: compute market trends and identify buy signals

- [ ] **3.1** Daily market stats aggregation (median/avg/min/max price per model+year)
- [ ] **3.2** Price trend calculation (30-day rolling average)
- [ ] **3.3** Individual listing price change tracking
- [ ] **3.4** Buy signal: price drops X% below rolling average
- [ ] **3.5** CLI command: `olx-parser analyze` - show market stats
- [ ] **3.6** CLI command: `olx-parser signals` - show buy opportunities

## Phase 4: Reporting & Visualization
> Goal: actionable insights for buy/sell decisions

- [ ] **4.1** Terminal dashboard with rich (table of watchlist models + trends)
- [ ] **4.2** Price history charts per model (matplotlib -> PNG)
- [ ] **4.3** CSV/JSON export for external analysis
- [ ] **4.4** Optional: simple web dashboard (FastAPI + htmx or Streamlit)

## Phase 5: Advanced
> Goal: improve data quality and decision accuracy

- [ ] **5.1** Currency normalization (UAH->USD via NBU API)
- [ ] **5.2** Outlier detection (suspiciously cheap/expensive listings)
- [ ] **5.3** Seasonal pattern analysis
- [ ] **5.4** Proxy rotation for resilient scraping
- [ ] **5.5** Telegram bot notifications for buy signals
