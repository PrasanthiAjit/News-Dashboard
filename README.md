# Tech & Science Headlines Dashboard

This folder contains a Streamlit dashboard that aggregates technology and science headlines from multiple RSS feeds.

Files:
- `app.py` - the Streamlit app
- `requirements.txt` - dependencies for this workspace
- `run.sh` - convenience launcher (uses repo `.venv` if present)

Quick start:

1. From the repo root (recommended), create/activate your virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tech_science_dashboard/requirements.txt
```

2. Run the dashboard:

```bash
./tech_science_dashboard/run.sh
```

or

```bash
streamlit run tech_science_dashboard/app.py
```

Notes:
- The app caches feeds for 10 minutes to avoid excessive network requests.
- Add or remove RSS sources by editing the `RSS_FEEDS` dict in `app.py`.
