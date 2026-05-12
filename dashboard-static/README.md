# Dashboard — stlite bundle for Cloudflare Pages

The browser-side dashboard that ships at the `nikit34/olx-car-parser` Pages
URL (and your custom domain, if connected). Source lives in `src/dashboard/`;
this directory is the **static bundle** stlite serves to visitors.

## How it boots

```
visitor → dashboard-static/index.html
            │
            ├─ <script type="module"> calls stlite.mount(...)
            │     ├─ fetches ./files/dashboard/*.py     (this repo's sources)
            │     ├─ fetches ./files/src/analytics/*.py (transitive imports)
            │     └─ fetches github.com/.../releases/download/latest-data/*.parquet
            │
            └─ stlite spins up Pyodide → installs pandas/plotly/pyarrow/numpy
              → mounts all files into MEMFS → runs 🔥_Recommendations.py
```

## Local development

```sh
# 1. Rebuild dashboard witnesses (parquets) from the local SQLite
python scripts/build_dashboard_data.py

# 2. Build the stlite bundle with local data bundled in
python scripts/build_stlite_bundle.py --include-data

# 3. Serve dashboard-static/ on localhost
python -m http.server 8600 --directory dashboard-static
# → http://localhost:8600  (index.html auto-routes parquet URLs to ./files/data/)
```

## Production deploy (Cloudflare Pages)

The connection is **manual one-time setup**; afterwards CF auto-builds on
every push to `master`.

1. Push your branch to GitHub.
2. Sign in to <https://dash.cloudflare.com/> → **Workers & Pages** → **Create
   application** → **Pages** → **Connect to Git**.
3. Pick the `olx-car-parser` repo. Build configuration:
   - **Framework preset:** *None*
   - **Build command:** `python3 scripts/build_stlite_bundle.py`
   - **Build output directory:** `dashboard-static`
   - **Root directory (Advanced):** `/` (leave default)
   - **Environment variables → Build → Add:**
     `PYTHON_VERSION` = `3.11`
4. Save and Deploy. First build takes ~2 min. The default URL is
   `<project-name>.pages.dev`; add a custom domain in **Custom domains** if
   you want.

That's it — every push to `master` triggers a new build automatically.

### Data refresh cadence

The dashboard fetches witness parquets from the `latest-data` GitHub Release
at browser mount time. `scrape-ci` re-uploads those files at the end of
every scrape run (see `.github/workflows/scrape.yml` → "Build dashboard
witnesses" + the upload step). GitHub's CDN caches them with `ETag` headers
so the browser sees fresh data within minutes of CI finishing.

You do **not** need to redeploy CF Pages when data refreshes — only when
dashboard code or layout changes.
