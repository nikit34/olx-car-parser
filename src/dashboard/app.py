"""Streamlit Cloud entrypoint — navigation router.

Streamlit's default sidebar labels come from the file name, which
gave us "app / Market Direction / Model Details" — ugly. Switch to
the ``st.navigation`` API so each page can declare its own friendly
title and icon. The page bodies live in:

  src/dashboard/_recommendations_page.py
  src/dashboard/pages/2_Market_Direction.py
  src/dashboard/pages/3_Model_Details.py

Streamlit Cloud's "Main file path" still points at this file
(src/dashboard/app.py), so no deploy-config change is needed.
``st.navigation`` overrides the auto-discovery of ``pages/``, hence
the explicit registration.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_here))
_sys.path.insert(0, str(_here.parent.parent))

import streamlit as st


st.set_page_config(
    page_title="OLX Car Deals",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(
        str(_here / "_recommendations_page.py"),
        title="Recommendations",
        icon="🔥",
        default=True,
    ),
    st.Page(
        str(_here / "pages" / "2_Market_Direction.py"),
        title="Market Direction",
        icon="📈",
    ),
    st.Page(
        str(_here / "pages" / "3_Model_Details.py"),
        title="Model Details",
        icon="🔍",
    ),
]

st.navigation(pages).run()
