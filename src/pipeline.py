from __future__ import annotations

import argparse
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.collect_news import INGESTED_PATH, collect_news, list_feed_profiles
from src.data.load_data import list_available_datasets, load_dataset
from src.llm.classify_news import NewsClassifier
from src.llm.summarize_news import NewsSummarizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "data" / "result.csv"
PLOT_DIR = PROJECT_ROOT / "outputs"
SOURCE_MODES = ("dataset", "ingest")

SOURCE_COLUMN_CANDIDATES = {
    "title": ["title", "Title", "headline"],
    "content": ["content", "Content", "summary", "description"],
    "industry": [
        "industry",
        "Industry",
        "Tag",
        "tag",
        "category",
        "Category",
        "feed_profile",
        "source",
    ],
    "source_name": ["source_name", "source", "publisher"],
    "source_url": ["source_url", "url", "link"],
    "published_at": ["published_at", "datetime", "published"],
    "collection_method": ["collection_method"],
    "feed_profile": ["feed_profile"],
    "collected_at": ["collected_at"],
}

INDUSTRY_KEYWORDS = {
    "Finance": [
        "bank",
        "bond",
        "earnings",
        "economy",
        "finance",
        "inflation",
        "interest rate",
        "market",
        "stock",
        "trading",
    ],
    "Manufacturing": [
        "assembly",
        "automotive",
        "factory",
        "industrial",
        "manufacturing",
        "plant",
        "production",
        "semiconductor",
    ],
    "Supply Chain": [
        "delivery",
        "export",
        "freight",
        "import",
        "inventory",
        "logistics",
        "shipment",
        "shipping",
        "supplier",
        "supply chain",
        "warehouse",
    ],
    "Technology": [
        "ai",
        "automation",
        "chip",
        "cloud",
        "cybersecurity",
        "data center",
        "digital",
        "platform",
        "software",
        "technology",
    ],
    "Policy": [
        "antitrust",
        "central bank",
        "fed",
        "government",
        "law",
        "policy",
        "regulation",
        "tariff",
        "tax",
    ],
}

STOPWORDS = {
    "about",
    "after",
    "amid",
    "also",
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
    "here",
    "hour",
    "hours",
    "into",
    "its",
    "day",
    "days",
    "latest",
    "month",
    "months",
    "more",
    "news",
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
    "watch",
    "were",
    "will",
    "with",
    "year",
    "years",
    "yesterday",
    "states",
}


@dataclass(slots=True)
class PipelineConfig:
    source_mode: str = "dataset"
    dataset_name: str = "financial"
    industry_profile: str = "mixed"
    prefer_live_collection: bool = True
    max_rows: int = 100
    sampling_strategy: str = "sample"
    random_state: int = 42
    output_path: Path = RESULT_PATH
    ingested_output_path: Path = INGESTED_PATH
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    force_fallback: bool = False


def list_source_modes() -> tuple[str, ...]:
    return SOURCE_MODES


def _find_first_matching_column(
    df: pd.DataFrame, candidates: list[str]
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("??", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_text_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column_name = _find_first_matching_column(df, candidates)
    if column_name is None:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    return df[column_name].fillna("").map(_clean_text)


def normalize_industry_label(raw_label: str, title: str, content: str) -> str:
    combined_text = " ".join([raw_label, title, content]).lower()
    scores = {industry: 0 for industry in INDUSTRY_KEYWORDS}

    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                scores[industry] += 1

    best_industry = max(scores, key=scores.get)
    if scores[best_industry] == 0:
        return "General Business"
    return best_industry


def normalize_source_dataframe(
    raw_df: pd.DataFrame,
    *,
    source_label: str,
    max_rows: int | None = 100,
    sampling_strategy: str = "sample",
    random_state: int = 42,
) -> pd.DataFrame:
    normalized_df = pd.DataFrame(
        {
            "title": _build_text_series(raw_df, SOURCE_COLUMN_CANDIDATES["title"]),
            "content": _build_text_series(raw_df, SOURCE_COLUMN_CANDIDATES["content"]),
        }
    )
    raw_industry = _build_text_series(raw_df, SOURCE_COLUMN_CANDIDATES["industry"])
    normalized_df["title"] = normalized_df.apply(
        lambda row: row["title"]
        or textwrap.shorten(row["content"], width=90, placeholder="..."),
        axis=1,
    )
    normalized_df["content"] = normalized_df.apply(
        lambda row: row["content"] or row["title"],
        axis=1,
    )
    normalized_df["industry"] = [
        normalize_industry_label(industry, title, content)
        for industry, title, content in zip(
            raw_industry.tolist(),
            normalized_df["title"].tolist(),
            normalized_df["content"].tolist(),
        )
    ]

    for metadata_column in [
        "source_name",
        "source_url",
        "published_at",
        "collection_method",
        "feed_profile",
        "collected_at",
    ]:
        normalized_df[metadata_column] = _build_text_series(
            raw_df,
            SOURCE_COLUMN_CANDIDATES[metadata_column],
        )

    normalized_df["source_name"] = normalized_df["source_name"].replace("", source_label)
    normalized_df["collection_method"] = normalized_df["collection_method"].replace(
        "",
        "csv_load",
    )
    normalized_df["feed_profile"] = normalized_df["feed_profile"].replace(
        "",
        source_label,
    )
    normalized_df["collected_at"] = normalized_df["collected_at"].replace(
        "",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    normalized_df = normalized_df[
        (normalized_df["title"].str.len() > 0)
        & (normalized_df["content"].str.len() > 0)
    ]
    normalized_df = normalized_df.drop_duplicates(subset=["title", "content"])

    if max_rows is not None and len(normalized_df) > max_rows:
        if sampling_strategy == "sample":
            normalized_df = normalized_df.sample(
                n=max_rows,
                random_state=random_state,
            )
        elif sampling_strategy == "head":
            normalized_df = normalized_df.head(max_rows)
        else:
            raise ValueError(
                "sampling_strategy must be either 'sample' or 'head'."
            )

    normalized_df = normalized_df.reset_index(drop=True)
    normalized_df.insert(0, "record_id", range(1, len(normalized_df) + 1))
    normalized_df.insert(1, "source_dataset", source_label)
    return normalized_df


def build_source_dataset(config: PipelineConfig) -> pd.DataFrame:
    if config.source_mode == "dataset":
        raw_df = load_dataset(config.dataset_name)
        source_label = config.dataset_name
    elif config.source_mode == "ingest":
        raw_df = collect_news(
            profile=config.industry_profile,
            max_articles=config.max_rows,
            prefer_live=config.prefer_live_collection,
            allow_fallback=True,
            output_path=config.ingested_output_path,
        )
        source_label = "automated_ingestion"
    else:
        available = ", ".join(SOURCE_MODES)
        raise ValueError(f"Unknown source_mode '{config.source_mode}'. Available: {available}")

    source_df = normalize_source_dataframe(
        raw_df,
        source_label=source_label,
        max_rows=config.max_rows,
        sampling_strategy=config.sampling_strategy,
        random_state=config.random_state,
    )
    source_df["source_mode"] = config.source_mode
    source_df["industry_profile"] = (
        config.industry_profile if config.source_mode == "ingest" else "local_dataset"
    )
    return source_df


def run_llm_analysis(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    force_fallback: bool = False,
) -> pd.DataFrame:
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    classifier = NewsClassifier(
        model=selected_model,
        force_fallback=force_fallback,
    )
    summarizer = NewsSummarizer(
        model=selected_model,
        force_fallback=force_fallback,
    )
    processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    enriched_rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        classification = classifier.classify(
            title=row_dict["title"],
            content=row_dict["content"],
            industry=row_dict["industry"],
        )
        summary = summarizer.summarize(
            title=row_dict["title"],
            content=row_dict["content"],
            category=classification.category,
            industry=row_dict["industry"],
        )

        row_dict["category"] = classification.category
        row_dict["summary"] = summary.summary
        row_dict["classification_method"] = classification.method
        row_dict["summary_method"] = summary.method
        row_dict["processed_at"] = processed_at
        row_dict["pipeline_stage"] = "classified_and_summarized"
        enriched_rows.append(row_dict)

    return pd.DataFrame(enriched_rows)


def save_result_dataset(
    df: pd.DataFrame,
    output_path: Path = RESULT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_news_pipeline(config: PipelineConfig | None = None) -> pd.DataFrame:
    config = config or PipelineConfig()
    source_df = build_source_dataset(config)
    result_df = run_llm_analysis(
        source_df,
        model=config.model,
        force_fallback=config.force_fallback,
    )
    save_result_dataset(result_df, output_path=config.output_path)
    return result_df


def load_result_dataset(output_path: Path = RESULT_PATH) -> pd.DataFrame:
    return pd.read_csv(output_path)


def ensure_result_dataset(
    config: PipelineConfig | None = None,
    output_path: Path = RESULT_PATH,
) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path)

    config = config or PipelineConfig(output_path=output_path)
    config.output_path = output_path
    return run_news_pipeline(config)


def get_category_distribution(df: pd.DataFrame) -> pd.Series:
    return df["category"].value_counts()


def get_industry_category_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["industry"], df["category"])


def get_collection_breakdown(df: pd.DataFrame) -> pd.Series:
    if "collection_method" not in df.columns:
        return pd.Series(dtype="int64")
    return df["collection_method"].value_counts()


def get_top_keywords(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    text_columns: list[pd.Series] = []
    for column in ["title", "summary", "content"]:
        if column in df.columns:
            text_columns.append(df[column].fillna("").astype(str))

    if not text_columns:
        return pd.DataFrame(columns=["keyword", "count"])

    combined_text = " ".join(" ".join(series.tolist()) for series in text_columns)
    words = re.findall(r"[a-z][a-z'-]+", combined_text.lower())
    filtered_words = [
        word for word in words if len(word) > 2 and word not in STOPWORDS
    ]
    keyword_counts = Counter(filtered_words).most_common(limit)
    return pd.DataFrame(keyword_counts, columns=["keyword", "count"])


def save_category_distribution_chart(
    category_distribution: pd.Series,
    output_path: Path | None = None,
) -> Path | None:
    if output_path is None:
        output_path = PLOT_DIR / "category_distribution.png"

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    category_distribution.plot(kind="bar", color="#295C77")
    plt.title("News Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_day2_analysis(
    config: PipelineConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    config = config or PipelineConfig()
    df = ensure_result_dataset(config=config, output_path=config.output_path)

    category_distribution = get_category_distribution(df)
    industry_category_table = get_industry_category_table(df)
    top_keywords = get_top_keywords(df)
    collection_breakdown = get_collection_breakdown(df)

    return {
        "dataset": df,
        "category_distribution": category_distribution,
        "industry_category_table": industry_category_table,
        "top_keywords": top_keywords,
        "collection_breakdown": collection_breakdown,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Industry Insight Agent pipeline end-to-end."
    )
    parser.add_argument(
        "--source-mode",
        default="dataset",
        choices=list_source_modes(),
        help="Choose between a saved CSV dataset and automated ingestion.",
    )
    parser.add_argument(
        "--dataset",
        default="financial",
        choices=list_available_datasets(raw_only=True),
        help="Saved CSV dataset to analyze when source-mode is dataset.",
    )
    parser.add_argument(
        "--profile",
        default="mixed",
        choices=list_feed_profiles(),
        help="Industry profile for automated ingestion.",
    )
    parser.add_argument(
        "--local-seed-only",
        action="store_true",
        help="Skip live RSS requests and build the ingestion dataset from local snapshots only.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum number of news rows to process.",
    )
    parser.add_argument(
        "--sampling",
        default="sample",
        choices=["sample", "head"],
        help="How to choose rows when the dataset is larger than max-rows.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULT_PATH),
        help="Path where the enriched CSV should be saved.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name to use when OPENAI_API_KEY is available.",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip OpenAI calls and use the rule-based pipeline only.",
    )
    return parser


def main() -> None:
    args = build_cli_parser().parse_args()
    config = PipelineConfig(
        source_mode=args.source_mode,
        dataset_name=args.dataset,
        industry_profile=args.profile,
        prefer_live_collection=not args.local_seed_only,
        max_rows=args.max_rows,
        sampling_strategy=args.sampling,
        output_path=Path(args.output),
        model=args.model,
        force_fallback=args.fallback_only,
    )
    result_df = run_news_pipeline(config)
    category_distribution = get_category_distribution(result_df)
    industry_category_table = get_industry_category_table(result_df)
    top_keywords = get_top_keywords(result_df)
    collection_breakdown = get_collection_breakdown(result_df)
    chart_path = save_category_distribution_chart(category_distribution)

    print("Saved result dataset to:", config.output_path)
    print("Processed rows:", len(result_df))
    print("Source mode:", config.source_mode)
    if config.source_mode == "dataset":
        print("Source dataset:", config.dataset_name)
    else:
        print("Ingestion profile:", config.industry_profile)
        print("Live collection enabled:", config.prefer_live_collection)
    print(
        "Classification methods:",
        result_df["classification_method"].value_counts().to_dict(),
    )
    print("Summary methods:", result_df["summary_method"].value_counts().to_dict())
    if not collection_breakdown.empty:
        print("Collection methods:", collection_breakdown.to_dict())
    print("\nCategory distribution:")
    print(category_distribution.to_string())
    print("\nIndustry/category table:")
    print(industry_category_table.to_string())
    print("\nTop keywords:")
    print(top_keywords.to_string(index=False))

    if chart_path is not None:
        print("\nSaved chart to:", chart_path)
    else:
        print("\nChart export skipped because matplotlib is not installed.")


if __name__ == "__main__":
    main()
