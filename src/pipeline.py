from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd

from src.data.load_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "data" / "result.csv"
PLOT_DIR = PROJECT_ROOT / "outputs"

ANALYSIS_COLUMN_CANDIDATES = {
    "title": ["title", "Title", "headline"],
    "industry": ["industry", "Industry", "Tag", "tag"],
    "content": ["content", "Content", "summary"],
    "category": ["category", "Category"],
}

STOPWORDS = {
    "about",
    "after",
    "amid",
    "and",
    "are",
    "been",
    "before",
    "being",
    "between",
    "from",
    "for",
    "had",
    "has",
    "have",
    "hour",
    "hours",
    "into",
    "its",
    "day",
    "days",
    "month",
    "months",
    "more",
    "said",
    "since",
    "that",
    "the",
    "their",
    "them",
    "than",
    "they",
    "this",
    "today",
    "tomorrow",
    "united",
    "week",
    "weeks",
    "was",
    "were",
    "will",
    "with",
    "year",
    "years",
    "yesterday",
    "states",
}


def _find_first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    available_columns = ", ".join(df.columns)
    raise ValueError(
        f"Could not find any of {candidates} in dataset columns: {available_columns}"
    )


def prepare_analysis_dataset(
    dataset_name: str = "financial_categorized",
) -> pd.DataFrame:
    raw_df = load_dataset(dataset_name)
    normalized_df = pd.DataFrame()

    for output_column, candidates in ANALYSIS_COLUMN_CANDIDATES.items():
        source_column = _find_first_matching_column(raw_df, candidates)
        normalized_df[output_column] = (
            raw_df[source_column].fillna("").astype(str).str.strip()
        )

    return normalized_df


def save_result_dataset(
    df: pd.DataFrame, output_path: Path = RESULT_PATH
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def ensure_result_dataset(output_path: Path = RESULT_PATH) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path)

    df = prepare_analysis_dataset()
    save_result_dataset(df, output_path=output_path)
    return df


def get_category_distribution(df: pd.DataFrame) -> pd.Series:
    return df["category"].value_counts()


def get_industry_category_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["industry"], df["category"])


def get_top_keywords(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    text = " ".join(df["content"].fillna("").astype(str).tolist()).lower()
    words = re.findall(r"[a-z][a-z'-]+", text)
    filtered_words = [
        word for word in words if len(word) > 2 and word not in STOPWORDS
    ]
    keyword_counts = Counter(filtered_words).most_common(limit)
    return pd.DataFrame(keyword_counts, columns=["keyword", "count"])


def save_category_distribution_chart(
    category_distribution: pd.Series, output_path: Path | None = None
) -> Path | None:
    if output_path is None:
        output_path = PLOT_DIR / "category_distribution.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    category_distribution.plot(kind="bar", color="#4C78A8")
    plt.title("News Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_day2_analysis() -> dict[str, pd.DataFrame | pd.Series]:
    df = prepare_analysis_dataset()
    save_result_dataset(df)

    category_distribution = get_category_distribution(df)
    industry_category_table = get_industry_category_table(df)
    top_keywords = get_top_keywords(df)

    return {
        "dataset": df,
        "category_distribution": category_distribution,
        "industry_category_table": industry_category_table,
        "top_keywords": top_keywords,
    }


def main() -> None:
    analysis = run_day2_analysis()
    chart_path = save_category_distribution_chart(analysis["category_distribution"])
    industry_category_preview = analysis["industry_category_table"].head(20)

    print("Saved analysis dataset to:", RESULT_PATH)
    print("\nCategory distribution:")
    print(analysis["category_distribution"].to_string())
    print("\nIndustry/category table (first 20 rows):")
    print(industry_category_preview.to_string())
    print("\nTop keywords:")
    print(analysis["top_keywords"].to_string(index=False))

    if chart_path is not None:
        print("\nSaved chart to:", chart_path)
    else:
        print("\nSkipped chart generation because matplotlib is not installed.")


if __name__ == "__main__":
    main()
