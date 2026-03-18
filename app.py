import streamlit as st

from src.data.load_data import list_available_datasets
from src.pipeline import (
    PipelineConfig,
    RESULT_PATH,
    ensure_result_dataset,
    get_category_distribution,
    get_collection_breakdown,
    get_industry_category_table,
    get_top_keywords,
    list_source_modes,
    run_news_pipeline,
)
from src.data.collect_news import list_feed_profiles


st.set_page_config(page_title="Industry Insight Agent", layout="wide")


def load_dashboard_dataset(config: PipelineConfig, refresh: bool):
    if refresh:
        return run_news_pipeline(config)
    return ensure_result_dataset(config=config, output_path=config.output_path)


st.title("Industry Insight Agent")
st.caption(
    "Automation-first pipeline for collecting, classifying, summarizing, and visualizing unstructured industry news"
)

available_datasets = list_available_datasets(raw_only=True)
source_modes = list_source_modes()
feed_profiles = list_feed_profiles()

with st.sidebar:
    st.header("Automation Settings")
    selected_source_mode = st.radio(
        "Input mode",
        options=source_modes,
        format_func=lambda value: "Automated Ingestion" if value == "ingest" else "Saved CSV Dataset",
    )

    selected_dataset = "financial"
    selected_profile = "mixed"
    prefer_live_collection = True

    if selected_source_mode == "dataset":
        selected_dataset = st.selectbox(
            "Source dataset",
            options=available_datasets,
            index=available_datasets.index("financial"),
        )
    else:
        selected_profile = st.selectbox(
            "Industry profile",
            options=feed_profiles,
            format_func=lambda value: value.title(),
        )
        prefer_live_collection = st.checkbox(
            "Try live RSS collection first",
            value=True,
            help="If unavailable, the pipeline can fall back to local seed data.",
        )

    max_rows = st.slider("Rows to process", min_value=25, max_value=300, value=100, step=25)
    sampling_strategy = st.radio(
        "Sampling strategy",
        options=["sample", "head"],
        index=0,
        horizontal=True,
    )
    force_fallback = st.checkbox(
        "Use rule-based AI fallback only",
        value=False,
        help="Enable this when OPENAI_API_KEY is not configured.",
    )
    refresh_requested = st.button("Run Automation Pipeline", use_container_width=True)

config = PipelineConfig(
    source_mode=selected_source_mode,
    dataset_name=selected_dataset,
    industry_profile=selected_profile,
    prefer_live_collection=prefer_live_collection,
    max_rows=max_rows,
    sampling_strategy=sampling_strategy,
    output_path=RESULT_PATH,
    force_fallback=force_fallback,
)

if refresh_requested or not RESULT_PATH.exists():
    with st.spinner("Running collection, classification, and summarization pipeline..."):
        df = load_dashboard_dataset(config, refresh=True)
else:
    df = load_dashboard_dataset(config, refresh=False)

category_distribution = get_category_distribution(df)
industry_category_table = get_industry_category_table(df)
top_keywords = get_top_keywords(df)
collection_breakdown = get_collection_breakdown(df)

current_source_mode = df["source_mode"].iat[0] if "source_mode" in df.columns else selected_source_mode
current_source = df["source_dataset"].iat[0] if "source_dataset" in df.columns else selected_dataset
current_profile = df["industry_profile"].iat[0] if "industry_profile" in df.columns else selected_profile
current_collection_method = (
    df["collection_method"].mode().iat[0] if "collection_method" in df.columns else "unknown"
)
classification_method = (
    df["classification_method"].mode().iat[0]
    if "classification_method" in df.columns
    else "unknown"
)
summary_method = (
    df["summary_method"].mode().iat[0]
    if "summary_method" in df.columns
    else "unknown"
)
processed_at = df["processed_at"].iat[0] if "processed_at" in df.columns else "N/A"

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Analyzed News", f"{len(df):,}")
metric_col2.metric("Source Mode", current_source_mode)
metric_col3.metric("Collection Method", current_collection_method)
metric_col4.metric("AI Engine", classification_method)

st.write(
    f"Result file: `{RESULT_PATH}`  |  Source: `{current_source}`  |  Profile: `{current_profile}`  |  Processed at: `{processed_at}`"
)

if not refresh_requested and (
    current_source_mode != selected_source_mode
    or (selected_source_mode == "dataset" and current_source != selected_dataset)
    or (selected_source_mode == "ingest" and current_profile != selected_profile)
):
    st.info("Click 'Run Automation Pipeline' to regenerate the result file with the current sidebar settings.")

st.subheader("Automation Summary")
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.write(
        "This dashboard shows how unstructured news moves through an automation workflow: ingestion, normalization, LLM analysis, structured storage, and trend analysis."
    )
with summary_col2:
    st.write(
        f"Classification mode: `{classification_method}`  |  Summary mode: `{summary_method}`"
    )

st.subheader("Result Table")
display_columns = [
    column
    for column in [
        "record_id",
        "title",
        "industry",
        "category",
        "summary",
        "source_name",
        "collection_method",
    ]
    if column in df.columns
]
st.dataframe(df[display_columns], use_container_width=True, hide_index=True)

chart_col, keyword_col = st.columns(2)

with chart_col:
    st.subheader("Category Distribution")
    st.bar_chart(category_distribution)

with keyword_col:
    st.subheader("Top Keywords")
    st.bar_chart(top_keywords.set_index("keyword"))

breakdown_col, table_col = st.columns(2)

with breakdown_col:
    st.subheader("Collection Breakdown")
    if collection_breakdown.empty:
        st.info("Collection metadata is not available for this result file.")
    else:
        st.bar_chart(collection_breakdown)

with table_col:
    st.subheader("Industry vs Category")
    st.dataframe(industry_category_table, use_container_width=True)

st.subheader("Keyword Detail")
st.dataframe(top_keywords, use_container_width=True, hide_index=True)
