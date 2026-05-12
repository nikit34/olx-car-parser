# Dashboard — stlite bundle for Cloudflare Pages

The browser-side dashboard that ships at the `nikit34/olx-car-parser` Pages
URL (and your custom domain, if connected). Source lives in `src/dashboard/`;
this directory is the **static bundle** stlite serves to visitors.

## How it boots

```
visitor → dashboard-static/index.html
            │
            ├─ <script type="module"> calls stlite.mount(...)
            │     ├─ fetches ./files/dashboard/*.py     (python sources, same-origin)
            │     ├─ fetches ./files/src/analytics/*.py (transitive imports, same-origin)
            │     └─ fetches ./data/dashboard/*.parquet (witnesses, same-origin)
            │
            └─ stlite spins up Pyodide → installs pandas/plotly/pyarrow/numpy
              → mounts all files into MEMFS → runs 🔥_Recommendations.py
```

Both code and data are served **same-origin** from CF Pages — Release
asset URLs don't return CORS headers, so direct browser fetch from a
cross-origin page fails. ``scripts/build_stlite_bundle.py`` materialises
both into ``dashboard-static/`` at build time.

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

   The default build behaviour fetches every witness parquet from the
   ``latest-data`` GitHub Release into ``dashboard-static/data/dashboard/``
   so they ship same-origin with the HTML.
4. Save and Deploy. First build takes ~2 min. The default URL is
   `<project-name>.pages.dev`; add a custom domain in **Custom domains** if
   you want.

That's it — every push to `master` triggers a new build automatically.

### Data refresh cadence

Witnesses are baked into the CF Pages deploy at build time, so a data
update requires a CF Pages rebuild. Two ways to trigger one:

1. **Manual:** open the deployment in CF Pages → "Retry deployment". Build
   downloads the current `latest-data` Release and ships it.
2. **Automatic (recommended):** create a CF Pages Deploy Hook in the
   project settings, copy the URL into a GH secret named
   `CF_PAGES_DEPLOY_HOOK`, then add a `curl -X POST` step to
   `scrape.yml` after the release upload. CF will rebuild within ~1 min
   of every scrape run.

(Same-origin serving is non-negotiable: GitHub Release assets don't
return `Access-Control-Allow-Origin`, so direct browser fetch fails
with `ERR_FAILED` even on public repos.)
