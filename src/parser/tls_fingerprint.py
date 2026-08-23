"""TLS handshake fingerprint for every client that talks to OLX.

Split out of the scraper so the listing-alive prober (src/storage) and the
photo fetcher share the exact same handshake — they hit the same CDN and
get blocked by the same rule.
"""

import ssl

import certifi


def build_ssl_context() -> ssl.SSLContext:
    """TLS context whose ClientHello does NOT carry httpx's default
    fingerprint.

    OLX sits behind CloudFront, and since 2026-08-10 its bot rules 403 the
    exact JA3/JA4 that httpx produces out of the box
    (``t13d1712h1_ab0a1bf427ad_8e6e362c5eac`` — plain
    ``ssl.create_default_context()`` plus ALPN, the single most common
    Python-bot handshake). The block is on the handshake, not on us: from
    the same machine, in the same minute, a browser, ``curl --http1.1``,
    ``wget``, ``requests`` and stdlib ``urllib`` all got 200 while httpx got
    403 on every HTTP version, every User-Agent and every header order.
    That is why the 2026-08 outage looked like "OLX changed its API" for
    thirteen days: the request never reached OLX.

    The two flags below are what ``urllib3`` (i.e. ``requests``) sets on top
    of the stdlib default. They change the extension list in the ClientHello
    — ``OP_NO_TICKET`` drops ``session_ticket``, ``post_handshake_auth`` adds
    its own extension — which moves us off the blocked fingerprint and onto
    the requests one: overwhelmingly common, legitimate traffic that a CDN
    cannot blanket-ban. Verified against the live API: default context 403,
    either flag alone 200.

    Do NOT "simplify" this back to ``httpx.Client(verify=True)``. That is the
    fingerprint that is banned.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.options |= ssl.OP_NO_TICKET
    ctx.post_handshake_auth = True
    return ctx


