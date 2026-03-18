import streamlit as st

from src.pipeline import (
    RESULT_PATH,
    ensure_result_dataset,
    get_category_distribution,
    get_industry_category_table,
    get_top_keywords,
)


st.set_page_config(page_title="Industry Insight Agent", layout="wide")

st.title("Industry Insight Agent")
st.caption("Day 2 dashboard for categorized industry news analysis")

if RESULT_PATH.exists():
    df = ensure_result_dataset()
else:
    with st.spinner("Preparing result dataset from the categorized news file..."):
        df = ensure_result_dataset()

category_distribution = get_category_distribution(df)
industry_category_table = get_industry_category_table(df)
top_keywords = get_top_keywords(df)

st.subheader("뉴스 데이터")
st.dataframe(df, use_container_width=True)

left_col, right_col = st.columns(2)
left_col.metric("Total News", f"{len(df):,}")
right_col.metric("Unique Categories", f"{df['category'].nunique():,}")

st.subheader("카테고리 분포")
st.bar_chart(category_distribution)

st.subheader("산업별 뉴스 비율")
st.dataframe(industry_category_table, use_container_width=True)

st.subheader("상위 키워드")
st.dataframe(top_keywords, use_container_width=True, hide_index=True)
