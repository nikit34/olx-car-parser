"""Point a request at the Cloudflare Worker relay instead of olx.pt directly.

OLX blocks the scrape host's address outright — every request from it comes
back 403 with a CloudFront page. The Worker fetches on our behalf from
Cloudflare's network and passes the upstream status and body through
untouched, which is what a liveness probe needs as much as a scrape does.

The prefixes mirror ``RELAY_PATH_PREFIXES`` in flipper-club/src/index.js:
the Worker refuses anything else, so sending a URL it will not accept just
earns the same 403 by a longer route.
"""
from __future__ import annotations

import os
import urllib.parse

RELAY_ORIGIN = "https://www.olx.pt"
RELAY_PATH_PREFIXES = ("/api/v1/offers", "/d/anuncio/")


def relay_config() -> tuple[str | None, str | None]:
    """The relay's URL and token, or (None, None) when it is not usable."""
    url = (os.environ.get("OLX_RELAY_URL") or "").strip() or None
    token = (os.environ.get("OLX_RELAY_TOKEN") or "").strip() or None
    if not url or not token:
        return None, None
    return url, token


def relay_rewrite(
    url: str,
    user_agent: str | None = None,
    relay_url: str | None = None,
    relay_token: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Return the URL to request and the extra headers it needs.

    Falls through unchanged when the relay is unconfigured or the target is
    not one the Worker will serve, so callers can route everything through
    here without deciding what is relayable.
    """
    if relay_url is None or relay_token is None:
        relay_url, relay_token = relay_config()
    if not relay_url or not relay_token or not url.startswith(RELAY_ORIGIN):
        return url, {}

    parts = urllib.parse.urlsplit(url)
    path_q = parts.path + (("?" + parts.query) if parts.query else "")
    if not any(path_q.startswith(prefix) for prefix in RELAY_PATH_PREFIXES):
        return url, {}

    target = f"{relay_url}?path={urllib.parse.quote(path_q, safe='')}"
    headers = {"X-Relay-Token": relay_token}
    if user_agent:
        headers["X-Relay-UA"] = user_agent
    return target, headers
