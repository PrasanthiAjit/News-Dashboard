import streamlit as st
import feedparser
import pandas as pd
import json
import io
import os
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

st.set_page_config(page_title="Global Tech & Science Headlines", layout="wide")
        # Display left column headline grid (cards side-by-side)
    n_cols = 3
        cols = st.columns(n_cols)
        for idx, row in df.iterrows():
            c = cols[idx % n_cols]
            with c:
                time_str = ''
                if row['published'] and row['published'].year > 1970:
                    time_str = row['published'].strftime('%Y-%m-%d %H:%M')
                title = row['title'] or ''
                summary = (row['summary'] or '')
                short = summary[:220] + ('...' if len(summary) > 220 else '')
                link = row.get('link') or '#'
                st.markdown(
                    f"<div class=\"nd-card\">\n  <div class=\"nd-card-title\">{title}</div>\n  <div class=\"nd-card-meta\">{row.get('source')} — {time_str}</div>\n  <div class=\"nd-card-summary\">{short}</div>\n  <div style=\"margin-top:8px;\"><a href=\"{link}\" target=\"_blank\">Open article</a></div>\n</div>",
                    unsafe_allow_html=True,
                )
    .nd-card-meta { color:#6c757d; font-size:12px; margin-bottom:8px; }
    .nd-card-summary { color:#333; font-size:14px; }
    .nd-card a { color:#1f77b4; text-decoration:none; }
    .nd-card:hover { box-shadow: 0 6px 20px rgba(31,119,180,0.08); }
</style>
""",
        unsafe_allow_html=True,
)

# List of RSS feeds (mix of global & specialized sources)
# Optimized: prioritize fast, reliable sources for quicker initial load
DEFAULT_RSS_FEEDS: Dict[str, str] = {
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    "TechCrunch": "http://feeds.feedburner.com/TechCrunch/",
}

# Default categories for built-in feeds
DEFAULT_RSS_FEEDS_CATEGORIES: Dict[str, List[str]] = {
    "BBC Technology": ["technical"],
    "BBC Science": ["science"],
    "The Verge": ["technical"],
    "Ars Technica": ["technical"],
    "TechCrunch": ["technical"],
}

# Path to persist user-provided feeds (optional)
RSS_FEEDS_PATH = os.path.join(os.path.dirname(__file__), "rss_feeds.json")

# Load persisted external feeds if present
def load_persisted_feeds(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # allow either dict or list of {name,url}
        # return raw structure; value can be a string (url) or an object {url, categories}
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # convert list of objects to dict-like structure
            out = {}
            for item in data:
                if isinstance(item, dict) and "name" in item and "url" in item:
                    out[item["name"]] = {"url": item["url"], "categories": item.get("categories", [])}
            return out
    except Exception:
        return {}
    return {}


# Merge default + external (persisted + runtime additions)
def build_feeds(defaults: Dict[str, str], external: Dict[str, str]) -> Dict[str, str]:
    feeds = defaults.copy()
    feeds.update(external)
    return feeds


def _normalize_text(s: str) -> str:
    """Normalize title text for fuzzy comparison: lower, remove punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove punctuation and non-word chars
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def deduplicate_df(df, threshold: float = 0.85):
    """Merge rows with similar titles across different sources.

    - Clusters titles using SequenceMatcher ratio >= threshold.
    - Keeps the most recent published item as representative and aggregates source names.
    """
    if df.empty:
        return df

    titles = df['title'].fillna('').tolist()
    norm_titles = [_normalize_text(t) for t in titles]

    clusters = []  # list of lists of row indices

    for idx, nt in enumerate(norm_titles):
        if not nt:
            # put empty titles into their own cluster
            clusters.append([idx])
            continue
        placed = False
        for cl in clusters:
            # compare to representative of cluster (first member)
            rep_idx = cl[0]
            rep_nt = norm_titles[rep_idx]
            if not rep_nt:
                continue
            ratio = difflib.SequenceMatcher(None, nt, rep_nt).ratio()
            if ratio >= threshold:
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    rows = []
    for cl in clusters:
        if len(cl) == 1:
            row = df.iloc[cl[0]].to_dict()
            # ensure aggregated links present for singletons
            row['all_links'] = [row.get('link')] if row.get('link') else []
            # structured sources list for consistent UI
            row['sources_list'] = [{'name': row.get('source'), 'link': row.get('link')}]
            rows.append(row)
        else:
            # pick the most recent published entry as representative
            subset = df.iloc[cl].copy()
            subset['published'] = pd.to_datetime(subset['published'], errors='coerce')
            subset = subset.sort_values('published', ascending=False)
            rep = subset.iloc[0].to_dict()
            # aggregate sources and links
            sources = list(dict.fromkeys(subset['source'].astype(str).tolist()))
            links = list(dict.fromkeys(subset['link'].astype(str).tolist()))
            rep['source'] = '; '.join(sources)
            # keep representative link but expose aggregated links if needed
            rep['all_links'] = links
            # build ordered list of unique (source, link) preserving subset order
            seen = set()
            src_list = []
            for _, r in subset.iterrows():
                name = str(r.get('source'))
                link = r.get('link')
                key = (name, link)
                if key in seen:
                    continue
                seen.add(key)
                src_list.append({'name': name, 'link': link})
            rep['sources_list'] = src_list
            rows.append(rep)

    out = pd.DataFrame(rows)
    # Ensure published is datetime and sort
    if 'published' in out.columns:
        out['published'] = pd.to_datetime(out['published'], errors='coerce').fillna(pd.Timestamp(0))
    out = out.sort_values('published', ascending=False).reset_index(drop=True)
    return out

@st.cache_data(ttl=1800)
def fetch_feed(url: str):
    """Fetch an RSS feed using requests (with timeout) and return parsed feed object.

    Using requests gives us explicit control over timeouts and headers and
    avoids passing unsupported keyword arguments to feedparser.parse.
    """
    import requests

    headers = {'User-Agent': 'Mozilla/5.0 (News-Dashboard)'}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        return parsed
    except Exception:
        return None

def _process_feed_entries(name: str, url: str):
    """Process a single feed and return list of entries."""
    parsed = fetch_feed(url)
    if not parsed or not hasattr(parsed, 'entries'):
        return []
    entries = []
    for e in parsed.entries:
        title = e.get('title', '')
        link = e.get('link', '')
        summary = e.get('summary', '') or e.get('description', '') or ''
        published = e.get('published') or e.get('updated') or ''
        published_parsed = None
        try:
            if 'published_parsed' in e and e.published_parsed:
                published_parsed = datetime(*e.published_parsed[:6])
            elif published:
                published_parsed = datetime.fromisoformat(published)
        except Exception:
            published_parsed = None
        summary_text = BeautifulSoup(summary, "html.parser").get_text()
        entries.append({
            'source': name,
            'title': title,
            'link': link,
            'summary': summary_text,
            'published': published_parsed,
        })
    return entries

@st.cache_data(ttl=1800)
def fetch_all_entries(selected_sources: List[str], feeds_hash: str):
    """Fetch all entries from selected feeds using concurrent requests."""
    entries = []
    
    # Build list of feeds to fetch
    feeds_to_fetch = [(name, url) for name, url in RSS_FEEDS.items() 
                      if not selected_sources or name in selected_sources]
    
    # Fetch feeds concurrently
    if feeds_to_fetch:
        progress_placeholder = st.empty()
        with ThreadPoolExecutor(max_workers=min(5, len(feeds_to_fetch))) as executor:
            future_to_name = {}
            for name, url in feeds_to_fetch:
                future = executor.submit(_process_feed_entries, name, url)
                future_to_name[future] = name
            
            completed = 0
            for future in as_completed(future_to_name):
                completed += 1
                try:
                    result = future.result(timeout=15)
                    entries.extend(result)
                except Exception:
                    pass  # silently skip failed feeds
                
                # Update progress
                progress_placeholder.info(f"Loading... {completed}/{len(feeds_to_fetch)} feeds fetched")
        
        progress_placeholder.empty()
    
    if not entries:
        return pd.DataFrame(columns=['source', 'title', 'link', 'summary', 'published'])
    df = pd.DataFrame(entries)
    # Deduplicate by link and title
    df = df.drop_duplicates(subset=['link', 'title'])
    # Ensure published is datetime; fill missing with epoch
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    df['published'] = df['published'].fillna(pd.Timestamp(0))
    df = df.sort_values('published', ascending=False).reset_index(drop=True)
    return df

# Helper: parse uploaded file (JSON or CSV)
def parse_uploaded_file(uploaded) -> Dict[str, str]:
    try:
        content = uploaded.getvalue()
        # Try JSON
        try:
            data = json.loads(content.decode("utf-8")) if isinstance(content, (bytes, bytearray)) else json.loads(content)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {item.get("name"): item.get("url") for item in data if isinstance(item, dict) and item.get("name") and item.get("url")}
        except Exception:
            pass
        # Try CSV (simple split)
        text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        out = {}
        for ln in lines:
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                out[name] = url
        return out
    except Exception:
        return {}


# Sidebar controls
st.sidebar.header("Filters & Options")

# Load persisted external feeds and prepare runtime state
persisted_feeds_raw = load_persisted_feeds(RSS_FEEDS_PATH)
# normalize persisted feeds into urls and categories
persisted_feeds: Dict[str, str] = {}
persisted_feeds_categories: Dict[str, List[str]] = {}
for name, v in (persisted_feeds_raw or {}).items():
    if isinstance(v, str):
        persisted_feeds[name] = v
    elif isinstance(v, dict):
        url = v.get("url")
        cats = v.get("categories") or v.get("category") or []
        if url:
            persisted_feeds[name] = url
        if cats:
            persisted_feeds_categories[name] = [c.strip().lower() for c in cats]

runtime_feeds: Dict[str, str] = {}
runtime_feeds_categories: Dict[str, List[str]] = {}

st.sidebar.subheader("Manage sources")
uploaded = st.sidebar.file_uploader("Upload sources (JSON or CSV)", type=["json", "csv", "txt"])
if uploaded is not None:
    parsed = parse_uploaded_file(uploaded)
    if parsed:
        # parsed values may be strings (url) or dicts {url, categories}
        for name, val in parsed.items():
            if isinstance(val, str):
                runtime_feeds[name] = val
            elif isinstance(val, dict):
                url = val.get("url") or val.get("link")
                cats = val.get("categories") or val.get("category") or []
                if url:
                    runtime_feeds[name] = url
                if cats:
                    runtime_feeds_categories[name] = [c.strip().lower() for c in cats]
        st.sidebar.success(f"Loaded {len(parsed)} sources from upload")

# UI to add a single source
new_name = st.sidebar.text_input("New source name")
new_url = st.sidebar.text_input("New source URL")
new_categories = st.sidebar.text_input("Categories (comma-separated)")
if st.sidebar.button("Add source"):
    if new_name and new_url:
        runtime_feeds[new_name] = new_url
        if new_categories:
            runtime_feeds_categories[new_name] = [c.strip().lower() for c in new_categories.split(",") if c.strip()]
        st.sidebar.success(f"Added source: {new_name}")
    else:
        st.sidebar.error("Provide both name and URL to add a source.")

# Build final feeds dict
external_feeds = persisted_feeds.copy()
external_feeds.update(runtime_feeds)
# build categories map merging defaults, persisted and runtime categories
external_feeds_categories: Dict[str, List[str]] = {}
external_feeds_categories.update(DEFAULT_RSS_FEEDS_CATEGORIES)
external_feeds_categories.update(persisted_feeds_categories)
external_feeds_categories.update(runtime_feeds_categories)

RSS_FEEDS = build_feeds(DEFAULT_RSS_FEEDS, external_feeds)

all_sources = list(RSS_FEEDS.keys())

# Category selector
all_categories = sorted({c for cats in external_feeds_categories.values() for c in cats})

# Quick preset groupings for common selections
PRESET_CATEGORY_GROUPS = {
    "All": all_categories,
    "Technical & Science": ["technical", "science"],
    "Education & Kids": ["education", "kids"],
    "News & Politics": ["political"],
    "Health": ["health"],
    "Sports": ["sports"],
}

preset_options = ["Custom"] + list(PRESET_CATEGORY_GROUPS.keys())
preset_choice = st.sidebar.radio("Quick presets", options=preset_options, index=0)

if all_categories:
    if preset_choice != "Custom":
        preset_cats = [c for c in PRESET_CATEGORY_GROUPS[preset_choice] if c in all_categories]
        selected_categories = st.sidebar.multiselect("Filter by categories", all_categories, default=preset_cats)
    else:
        selected_categories = st.sidebar.multiselect("Filter by categories", all_categories, default=all_categories)
else:
    selected_categories = []

# compute default selected sources based on selected categories
if selected_categories:
    default_selected = [name for name, cats in external_feeds_categories.items() if any(c in selected_categories for c in cats) and name in all_sources]
    if not default_selected:
        default_selected = all_sources
else:
    default_selected = all_sources

selected = st.sidebar.multiselect("Select sources", all_sources, default=default_selected)
search = st.sidebar.text_input("Search (title / summary)")
limit = st.sidebar.number_input("Max headlines", min_value=10, max_value=500, value=100, step=10)
refresh = st.sidebar.button("Refresh now")

# Deduplication threshold control
st.sidebar.subheader("Deduplication")
if 'dedupe_threshold' not in st.session_state:
    st.session_state['dedupe_threshold'] = 0.85
threshold = st.sidebar.slider(
    "Headline similarity (higher = stricter dedupe)",
    min_value=0.50,
    max_value=0.99,
    value=st.session_state['dedupe_threshold'],
    step=0.01,
    help="Lower values merge more headlines as duplicates; higher values are stricter and merge only very similar headlines.",
)
st.session_state['dedupe_threshold'] = threshold

st.sidebar.markdown("---")
st.sidebar.markdown("About: This dashboard pulls headlines from public RSS feeds. Use the upload/add controls above to add external feeds. Click 'Save sources' to persist them to `rss_feeds.json`.")

# Optionally save the merged external feeds to disk (include categories)
if st.sidebar.button("Save sources"):
    try:
        items = []
        for name, url in external_feeds.items():
            cats = external_feeds_categories.get(name, [])
            items.append({"name": name, "url": url, "categories": cats})
        with open(RSS_FEEDS_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        st.sidebar.success(f"Saved {len(items)} sources to {RSS_FEEDS_PATH}")
    except Exception as e:
        st.sidebar.error(f"Failed to save sources: {e}")

# Fetch entries
feeds_hash = str(sorted(RSS_FEEDS.items()))
with st.spinner("Fetching feeds… (cached for 10 minutes)" if not refresh else "Refreshing feeds…"):
    raw_df = fetch_all_entries(selected, feeds_hash)
    original_count = 0 if raw_df is None else len(raw_df)
    # Deduplicate across different sources (fuzzy title matching)
    try:
        df = deduplicate_df(raw_df, threshold=threshold)
    except Exception:
        df = raw_df
    deduped_count = 0 if df is None else len(df)

if df.empty:
    st.info("No headlines found. Try selecting different sources or refresh.")
else:
    # Apply search filter
    if search:
        mask = df['title'].str.contains(search, case=False, na=False) | df['summary'].str.contains(search, case=False, na=False)
        df = df[mask]

    df = df.head(int(limit))

    # Layout: left column for list, right for details
    left, right = st.columns([1, 2])

    with left:
        st.subheader(f"Headlines ({len(df)})")
        # show counts before/after dedupe
        try:
            st.caption(f"Shown: {len(df)} (deduped) — original fetched: {original_count}")
        except Exception:
            pass

    # Create a selectbox for article selection (in right column at top)
    with right:
        st.subheader("Select article")
        # Build headline labels for the selectbox
        headline_labels = []
        for idx, row in df.iterrows():
            time_str = ''
            if row['published'] and row['published'].year > 1970:
                time_str = row['published'].strftime('%Y-%m-%d %H:%M')
            label = f"{row['title'][:60]}... ({row['source']}) — {time_str}"
            headline_labels.append(label)
        # Use query param to allow clickable card selection (e.g. ?selected=3)
        qparams = st.experimental_get_query_params()
        if 'selected' in qparams:
            try:
                qidx = int(qparams.get('selected', [0])[0])
                st.session_state['selected_idx'] = max(0, min(qidx, len(headline_labels) - 1))
            except Exception:
                pass

        # Ensure a selected index in session state
        if 'selected_idx' not in st.session_state:
            st.session_state['selected_idx'] = 0

        if headline_labels:
            # keep the selectbox in sync with session state
            current_index = min(st.session_state.get('selected_idx', 0), len(headline_labels) - 1)
            selected_label = st.selectbox("Choose an article:", options=headline_labels, index=current_index)
            sel_idx = headline_labels.index(selected_label)
            st.session_state['selected_idx'] = sel_idx
        else:
            sel_idx = st.session_state.get('selected_idx', 0)

    # Display left column headline grid (cards side-by-side)
    with left:
        # show counts (already displayed in header) and render cards
        n_cols = 3
        cols = st.columns(n_cols)
        for idx, row in df.iterrows():
            c = cols[idx % n_cols]
            with c:
                time_str = ''
                if row['published'] and row['published'].year > 1970:
                    time_str = row['published'].strftime('%Y-%m-%d %H:%M')
                title = row['title'] or ''
                summary = (row['summary'] or '')
                short = summary[:220] + ('...' if len(summary) > 220 else '')
                link = row.get('link') or '#'
                # Title becomes a link that sets the `selected` query param so clicking it selects the article
                select_href = f"?selected={idx}"
                st.markdown(
                    f"<div class=\"nd-card\">\n  <div class=\"nd-card-title\"><a href=\"{select_href}\">{title}</a></div>\n  <div class=\"nd-card-meta\">{row.get('source')} — {time_str}</div>\n  <div class=\"nd-card-summary\">{short}</div>\n  <div style=\"margin-top:8px;\"><a href=\"{link}\" target=\"_blank\">Open article</a></div>\n</div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("Article details")
        if not df.empty:
            row = df.iloc[int(sel_idx)]
            st.write(f"**{row['title']}**")
            # list all sources (name + link) when available
            sources_list = row.get('sources_list') or []
            if sources_list:
                st.markdown("**Sources:**")
                for s in sources_list:
                    name = s.get('name')
                    link = s.get('link')
                    if link:
                        st.markdown(f"- [{name}]({link})")
                    else:
                        st.markdown(f"- {name}")
            else:
                st.write(f"Source: {row['source']}")
            if row['published'] and row['published'].year > 1970:
                st.write(f"Published: {row['published'].strftime('%Y-%m-%d %H:%M')}")
            st.write(row['summary'])
            # show representative link
            if row.get('link'):
                st.markdown(f"[Read representative article]({row['link']})")
            # show aggregated links if available
            all_links = row.get('all_links') or []
            if all_links:
                if len(all_links) > 1:
                    st.markdown("**Other sources / links for this headline:**")
                for l in all_links:
                    if l:
                        st.markdown(f"- [Open]({l})")

st.markdown('---')
st.caption('Feeds: ' + ', '.join([k for k in RSS_FEEDS]))
