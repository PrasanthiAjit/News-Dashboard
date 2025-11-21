import streamlit as st
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Global Tech & Science Headlines", layout="wide")

st.title("Global Tech & Science Headlines")
st.markdown("Aggregated headlines from major tech & science news sources (RSS).")

# List of RSS feeds (mix of global & specialized sources)
RSS_FEEDS: Dict[str, str] = {
    "Google News - Technology (Global)": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
    "Google News - Science (Global)": "https://news.google.com/rss/search?q=science&hl=en-US&gl=US&ceid=US:en",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Wired": "https://www.wired.com/feed/rss",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    "Nature News": "https://www.nature.com/nature.rss",
    "Science Magazine": "https://www.sciencemag.org/rss/news_current.xml",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    "TechCrunch": "http://feeds.feedburner.com/TechCrunch/",
}

@st.cache_data(ttl=600)
def fetch_feed(url: str):
    """Fetch an RSS feed and return parsed feed object."""
    try:
        parsed = feedparser.parse(url)
        return parsed
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_all_entries(selected_sources: List[str]):
    entries = []
    for name, url in RSS_FEEDS.items():
        if selected_sources and name not in selected_sources:
            continue
        parsed = fetch_feed(url)
        if not parsed or not hasattr(parsed, 'entries'):
            continue
        for e in parsed.entries:
            title = e.get('title', '')
            link = e.get('link', '')
            summary = e.get('summary', '') or e.get('description', '') or ''
            # Try to get a published date
            published = e.get('published') or e.get('updated') or ''
            published_parsed = None
            try:
                if 'published_parsed' in e and e.published_parsed:
                    published_parsed = datetime(*e.published_parsed[:6])
                elif published:
                    published_parsed = datetime.fromisoformat(published)
            except Exception:
                published_parsed = None

            # Clean summary HTML
            summary_text = BeautifulSoup(summary, "html.parser").get_text()

            entries.append(
                {
                    'source': name,
                    'title': title,
                    'link': link,
                    'summary': summary_text,
                    'published': published_parsed,
                }
            )
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

# Sidebar controls
st.sidebar.header("Filters & Options")
all_sources = list(RSS_FEEDS.keys())
selected = st.sidebar.multiselect("Select sources", all_sources, default=all_sources)
search = st.sidebar.text_input("Search (title / summary)")
limit = st.sidebar.number_input("Max headlines", min_value=10, max_value=500, value=100, step=10)
refresh = st.sidebar.button("Refresh now")

st.sidebar.markdown("---")
st.sidebar.markdown("About: This dashboard pulls headlines from public RSS feeds. You can add additional feeds by editing the `RSS_FEEDS` dict in the source file.")

# Fetch entries
with st.spinner("Fetching feeds… (cached for 10 minutes)" if not refresh else "Refreshing feeds…"):
    df = fetch_all_entries(selected)

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
        # show a compact table of headlines
        for idx, row in df.iterrows():
            time_str = ''
            if row['published'] and row['published'].year > 1970:
                time_str = row['published'].strftime('%Y-%m-%d %H:%M')
            st.markdown(f"**{row['title']}**  \\n*{row['source']} — {time_str}*  \\n{row['summary'][:250]}{'...' if len(row['summary'])>250 else ''}")
            st.markdown(f"[Open article]({row['link']})")
            st.markdown('---')

    with right:
        st.subheader("Selected article")
        sel_idx = st.number_input("Article index (0-based)", min_value=0, max_value=max(0, len(df)-1), value=0)
        if not df.empty:
            row = df.iloc[int(sel_idx)]
            st.write(f"**{row['title']}**")
            st.write(f"Source: {row['source']}")
            if row['published'] and row['published'].year > 1970:
                st.write(f"Published: {row['published'].strftime('%Y-%m-%d %H:%M')}")
            st.write(row['summary'])
            st.markdown(f"[Read full article]({row['link']})")

st.markdown('---')
st.caption('Feeds: ' + ', '.join([k for k in RSS_FEEDS]))
