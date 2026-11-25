"""Central configuration for the PageSense backend.

Purpose of this module:

- Be the single home for environment-driven settings such as CORS origins,
  FastAPI host/port, and Weaviate connection details.
- Keep configuration import-light and side-effect free so any module can read
  settings without triggering network calls or model loading.
- Give later phases a stable import point (`app.config`) instead of scattering
  `os.getenv(...)` calls across route handlers, stores, and scripts.

Current state: settings are still read locally in `app/main.py` and
`app/store/weaviate_store.py`. This file is the documented placeholder for the
migration; no behavior changes until those modules import from here.
"""
