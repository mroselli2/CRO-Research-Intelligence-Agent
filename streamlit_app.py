"""
Read-only viewer over the compiled evidence table. Does not call Groq, GROBID, or
Europe PMC, and needs no API keys - it only reads a CSV. Safe to deploy as-is.
"""

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    (BASE_DIR / "data" / "compiled_article_evidence_table.csv", False),
    (BASE_DIR / "outputs" / "compiled_article_evidence_table.csv", True),
    (BASE_DIR / "sample_outputs" / "compiled_article_evidence_table.csv", True),
]

NOT_REPORTED = "Not reported"

# Light-mode chart chrome + a single restrained accent, pulled from the project's
# validated reference palette (light surfaces + categorical slot 1 "blue") rather than
# invented hex values - one accent used sparingly as chrome, not as data-identity color.
PAGE = "#ffffff"
SURFACE = "#ffffff"
SIDEBAR_SURFACE = "#d1d9e0"
BORDER = "rgba(11, 11, 11, 0.08)"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#0b0b0b"
TEXT_MUTED = "#426267"
ACCENT = "#2a78d6"

st.set_page_config(page_title="CRO Evidence Review Dashboard", page_icon="🧬", layout="wide")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {PAGE};
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stHeader"] {{
        background-color: transparent;
    }}
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_SURFACE};
        border-right: 1px solid {BORDER};
        min-width: 230px !important;
        max-width: 240px !important;
    }}
    [data-testid="stSidebar"] label {{
        font-size: 0.85rem;
    }}
    .block-container {{
        max-width: 1700px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }}
    h1 {{
        font-size: 2.4rem;
        font-weight: 700;
        border-bottom: 2px solid {ACCENT};
        padding-bottom: 0.5rem;
        margin-bottom: 0.3rem;
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {TEXT_SECONDARY};
    }}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
        font-size: 0.92rem;
    }}
    [data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-top: 2px solid {ACCENT};
        border-radius: 8px;
        padding: 0.7rem 0.9rem 0.6rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        font-size: 0.82rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {ACCENT};
        font-size: 1.6rem;
    }}
    hr {{
        border-color: {BORDER} !important;
    }}
    [data-testid="stDataFrame"] {{
        font-size: 0.82rem;
    }}
    a, a:visited {{
        color: {ACCENT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_missing(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NOT_REPORTED
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "not_reported", "not_applicable", "[]"):
        return NOT_REPORTED
    return text


def humanize_label(value) -> str:
    text = clean_missing(value)
    if text == NOT_REPORTED:
        return text
    text = text.replace("meta_analysis", "meta-analysis").replace("_", " ")
    return text.capitalize()


def parse_list_cell(value) -> str:
    """List-valued fields are stored in the CSV as Python repr strings; render them as text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NOT_REPORTED
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return "; ".join(str(item) for item in parsed) if parsed else NOT_REPORTED
    except (ValueError, SyntaxError, TypeError):
        pass
    return clean_missing(value)


TITLE_CASE_SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
    "on", "or", "the", "to", "vs", "via", "with",
}


def title_case(text: str) -> str:
    words = text.split(" ")
    result = []
    last_index = len(words) - 1
    starts_subtitle = False
    for i, word in enumerate(words):
        force = starts_subtitle
        starts_subtitle = word.endswith(":")
        if not word or any(c.isupper() for c in word):
            result.append(word)
            continue
        bare = word.strip(",:;()").lower()
        if not force and i not in (0, last_index) and bare in TITLE_CASE_SMALL_WORDS:
            result.append(word)
        else:
            result.append(word[0].upper() + word[1:])
    return " ".join(result)


def shorten(text, max_len: int) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_year(year) -> str:
    text = clean_missing(year)
    if text == NOT_REPORTED:
        return text
    try:
        return str(int(float(year)))
    except (TypeError, ValueError):
        return text


def format_journal_year(journal, year) -> str:
    journal_text = clean_missing(journal)
    year_text = format_year(year)
    if journal_text == NOT_REPORTED and year_text == NOT_REPORTED:
        return NOT_REPORTED
    if journal_text == NOT_REPORTED:
        return year_text
    if year_text == NOT_REPORTED:
        return journal_text
    return f"{journal_text} ({year_text})"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


csv_path, using_legacy_path = None, False
for candidate, is_legacy in CANDIDATE_PATHS:
    if candidate.exists():
        csv_path, using_legacy_path = candidate, is_legacy
        break

if csv_path is None:
    st.error("No evidence table found. Run cro_research_intelligence_agent.py first.")
    st.stop()

df = load_data(csv_path)

st.title("CRO Evidence Review Dashboard")
st.caption(
    "Review a compiled evidence table generated from a reproducible article-intake and "
    "extraction pipeline, with structured fields for study design, endpoints, biomarkers, "
    "and operational feasibility signals."
)

with st.sidebar:
    st.header("Filters")
    disease_areas = st.multiselect("Disease area", sorted(df["disease_area"].dropna().unique()),
                                    format_func=humanize_label)
    study_types = st.multiselect("Study type", sorted(df["study_type"].dropna().unique()),
                                  format_func=humanize_label)
    risk_levels = st.multiselect("Feasibility risk", sorted(df["operational_feasibility_risk"].dropna().unique()),
                                  format_func=humanize_label)
    review_levels = st.multiselect("Review priority", sorted(df["review_priority"].dropna().unique()),
                                    format_func=humanize_label)

filtered = df.copy()
for column, selected in [
    ("disease_area", disease_areas),
    ("study_type", study_types),
    ("operational_feasibility_risk", risk_levels),
    ("review_priority", review_levels),
]:
    if selected:
        filtered = filtered[filtered[column].isin(selected)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Articles", len(filtered))
col2.metric("High review priority", int((filtered["review_priority"] == "high").sum()))
col3.metric(
    "Avg. extraction quality",
    f"{filtered['extraction_quality_score'].mean():.0f}" if len(filtered) else "—",
)
col4.metric("High feasibility risk", int((filtered["operational_feasibility_risk"] == "high").sum()))

st.divider()

display_df = pd.DataFrame({
    "Title": filtered["title"].apply(lambda v: shorten(title_case(clean_missing(v)), 100)),
    "Disease area": filtered["disease_area"].apply(humanize_label),
    "Study type": filtered["study_type"].apply(humanize_label),
    "Primary endpoint": filtered["primary_endpoint"].apply(lambda v: shorten(clean_missing(v), 90)),
    "Feasibility score": filtered["operational_feasibility_risk"].apply(humanize_label),
    "Reasoning": filtered["feasibility_risk_reason"].apply(lambda v: shorten(clean_missing(v), 90)),
    "Visit schedule demands": filtered["visit_schedule_demands"].apply(lambda v: shorten(clean_missing(v), 70)),
    "Safety monitoring needs": filtered["safety_monitoring_needs"].apply(lambda v: shorten(clean_missing(v), 70)),
    "Site activation considerations": filtered["site_activation_considerations"].apply(lambda v: shorten(clean_missing(v), 70)),
    "Review priority": filtered["review_priority"].apply(humanize_label),
    "Quality score": filtered["extraction_quality_score"],
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Title": st.column_config.TextColumn("Title", width="large"),
        "Quality score": st.column_config.NumberColumn(
            "Quality score", format="%d", help="0-100 extraction confidence score",
        ),
    },
)

st.divider()
st.subheader("Article detail")

if len(filtered):
    def option_label(article_id: str) -> str:
        row = filtered.loc[filtered["article_id"] == article_id].iloc[0]
        short_title = shorten(title_case(clean_missing(row["title"])), 70)
        meta = format_journal_year(row.get("journal"), row.get("publication_year"))
        return short_title if meta == NOT_REPORTED else f"{short_title} — {meta}"

    id_to_label = {aid: option_label(aid) for aid in filtered["article_id"]}
    selected_id = st.selectbox(
        "Select an article",
        filtered["article_id"],
        format_func=lambda aid: id_to_label.get(aid, aid),
    )
    article = filtered[filtered["article_id"] == selected_id].iloc[0]

    st.markdown(f"### {title_case(clean_missing(article['title']))}")
    st.write(format_journal_year(article.get("journal"), article.get("publication_year")))

    doi = article.get("doi")
    if doi and str(doi) != "not_reported":
        st.write(f"DOI: [https://doi.org/{doi}](https://doi.org/{doi})")

    st.markdown("**Abstract**")
    st.caption(shorten(clean_missing(article.get("abstract")), 2500))

    pico_col, endpoint_col = st.columns(2)
    with pico_col:
        st.markdown("**PICO**")
        st.write(f"- Population: {clean_missing(article.get('population'))}")
        st.write(f"- Intervention: {clean_missing(article.get('intervention_type'))}")
        st.write(f"- Comparator: {clean_missing(article.get('comparator'))}")
    with endpoint_col:
        st.markdown("**Endpoints & biomarkers**")
        st.write(f"- Primary endpoint: {clean_missing(article.get('primary_endpoint'))}")
        st.write(f"- Secondary endpoints: {parse_list_cell(article.get('secondary_endpoints', ''))}")
        st.write(f"- Biomarkers: {parse_list_cell(article.get('biomarkers', ''))}")

    st.markdown("**Feasibility summary**")
    st.write(clean_missing(article.get("feasibility_summary")))

    # Length here is whatever the Groq prompt produced for plain_english_blurb - a
    # future pipeline change could ask for a longer summary; not expanded here.
    st.markdown("**Plain-language summary**")
    st.write(clean_missing(article.get("plain_english_blurb")))

    st.markdown("**Source excerpt**")
    st.caption(clean_missing(article.get("source_excerpt")))
else:
    st.info("No articles match the current filters.")
