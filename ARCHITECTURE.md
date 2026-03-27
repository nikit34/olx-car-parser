# OLX Car Parser - Architecture

## Goal
Collect private car listings from OLX, track price dynamics over time,
and identify optimal buy/sell moments for specific car models.

## High-Level Architecture

```
[OLX Website] --> [Parser/Scraper] --> [Storage (SQLite)] --> [Analytics Engine] --> [Dashboard/Reports]
                       |                      |
                  Scheduler (cron)      Price History DB
```

## Components

### 1. Parser (`src/parser/`)
- **scraper.py** - HTTP client + HTML parsing (requests + BeautifulSoup / httpx)
- **listing_parser.py** - Extract structured data from listing pages
- **pagination.py** - Handle search result pagination
- **anti_block.py** - Rate limiting, user-agent rotation, proxy support

Collects per listing:
- Title, price (UAH/USD), currency
- Brand, model, year, mileage, engine type/volume, transmission, body type
- City/region, seller type (private/dealer)
- Listing URL, OLX listing ID
- Date posted, date scraped

### 2. Models (`src/models/`)
- **listing.py** - SQLAlchemy/Pydantic model for a car listing
- **price_snapshot.py** - Price observation tied to a listing over time

### 3. Storage (`src/storage/`)
- **database.py** - SQLite connection + schema management (Alembic or raw migrations)
- **repository.py** - CRUD operations for listings and price snapshots
- Uses SQLite for simplicity; can migrate to PostgreSQL later

### 4. Analytics (`src/analytics/`)
- **price_tracker.py** - Track price changes per listing over time
- **market_analyzer.py** - Aggregate stats per model: median price, trend, volatility
- **signals.py** - Buy/sell signal generation (price below rolling average, etc.)
- **reports.py** - Generate summary reports (CSV, terminal, or HTML)

### 5. Utils (`src/utils/`)
- **config.py** - Load config from YAML/env
- **logger.py** - Structured logging

### 6. Config (`config/`)
- **settings.yaml** - Target URLs, scrape intervals, DB path, proxy list
- **models_watchlist.yaml** - List of car models to track with budget ranges

## Data Flow

1. **Scrape** - Scheduler triggers scraper every N hours
2. **Parse** - Raw HTML -> structured Listing objects
3. **Store** - Upsert listings; append price snapshots
4. **Analyze** - On-demand or scheduled: compute trends, generate signals
5. **Report** - Output recommendations: "Toyota Camry 2018 is 12% below 30-day avg"

## Tech Stack

| Layer       | Technology               |
|-------------|--------------------------|
| Language    | Python 3.11+             |
| HTTP Client | httpx (async support)    |
| HTML Parser | BeautifulSoup4 / lxml    |
| ORM         | SQLAlchemy 2.0           |
| Validation  | Pydantic v2              |
| DB          | SQLite (-> PostgreSQL)   |
| Scheduler   | APScheduler / cron       |
| Analytics   | pandas + matplotlib      |
| CLI         | typer / click            |

## Database Schema (simplified)

```sql
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    olx_id TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER,
    mileage_km INTEGER,
    engine_volume REAL,
    engine_type TEXT,        -- petrol, diesel, gas, electric, hybrid
    transmission TEXT,       -- manual, automatic
    body_type TEXT,          -- sedan, suv, hatchback, ...
    city TEXT,
    region TEXT,
    seller_type TEXT,        -- private, dealer
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE price_snapshots (
    id INTEGER PRIMARY KEY,
    listing_id INTEGER REFERENCES listings(id),
    price REAL NOT NULL,
    currency TEXT NOT NULL,  -- UAH, USD, EUR
    price_usd REAL,         -- normalized to USD
    scraped_at TIMESTAMP NOT NULL
);

CREATE TABLE market_stats (
    id INTEGER PRIMARY KEY,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    year_from INTEGER,
    year_to INTEGER,
    date DATE NOT NULL,
    median_price_usd REAL,
    avg_price_usd REAL,
    min_price_usd REAL,
    max_price_usd REAL,
    listing_count INTEGER,
    UNIQUE(brand, model, year_from, year_to, date)
);
```
