from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATA_FILES = {
    "financial": "Financial.csv",
    "financial_categorized": "Financial_Categorized.csv",
    "financial_sentiment": "Financial_Sentiment.csv",
    "financial_sentiment_categorized": "Financial_Sentiment_Categorized.csv",
    "google_daily_news": "Google_Daily_News.csv",
}

RAW_DATASET_NAMES = tuple(
    name for name in DATA_FILES if not name.endswith("_categorized")
)
CATEGORIZED_DATASET_NAMES = tuple(
    name for name in DATA_FILES if name.endswith("_categorized")
)

DISPLAY_COLUMN_CANDIDATES = {
    "title": ["title", "Title", "headline"],
    "content": ["content", "Content", "summary"],
    "industry": ["industry", "Industry", "Category", "category", "Tag", "tag"],
}


def get_data_path(name: str) -> Path:
    if name not in DATA_FILES:
        available_names = ", ".join(DATA_FILES)
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available_names}"
        )

    return DATA_DIR / DATA_FILES[name]


def list_available_datasets(
    *, raw_only: bool = False, categorized_only: bool = False
) -> tuple[str, ...]:
    if raw_only and categorized_only:
        raise ValueError("raw_only and categorized_only cannot both be True.")

    if raw_only:
        return RAW_DATASET_NAMES
    if categorized_only:
        return CATEGORIZED_DATASET_NAMES

    return tuple(DATA_FILES)


def load_dataset(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(get_data_path(name), **kwargs)


def load_all_datasets() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(DATA_DIR / filename)
        for name, filename in DATA_FILES.items()
    }


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str], dataset_name: str
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(df.columns.astype(str))
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}. "
            f"Available columns: {available}"
        )


def _find_first_matching_column(
    df: pd.DataFrame, candidates: list[str]
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def format_dataset_for_display(
    df: pd.DataFrame, max_rows: int = 5, text_width: int = 80
) -> pd.DataFrame:
    display_df = pd.DataFrame()

    for output_column, candidates in DISPLAY_COLUMN_CANDIDATES.items():
        source_column = _find_first_matching_column(df, candidates)

        if source_column is None:
            display_df[output_column] = ""
            continue

        display_df[output_column] = (
            df[source_column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.slice(0, text_width)
        )

    return display_df.head(max_rows)


def print_dataset_preview(name: str, max_rows: int = 5) -> None:
    df = load_dataset(name)
    preview_df = format_dataset_for_display(df, max_rows=max_rows)

    print(f"\n[{name}]")
    print(preview_df.to_string(index=False))


def print_all_dataset_previews(max_rows: int = 5) -> None:
    for name in DATA_FILES:
        print_dataset_preview(name, max_rows=max_rows)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    print_all_dataset_previews(max_rows=3)
