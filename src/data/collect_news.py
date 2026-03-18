from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
import re
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from src.data.load_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTED_PATH = PROJECT_ROOT / "data" / "ingested_news.csv"

PROFILE_KEYWORDS = {
    "finance": [
        "bank",
        "bond",
        "economy",
        "earnings",
        "finance",
        "inflation",
        "interest rate",
        "investment",
        "market",
        "stock",
        "trading",
    ],
    "manufacturing": [
        "assembly",
        "automotive",
        "battery",
        "factory",
        "industrial",
        "logistics",
        "manufacturing",
        "plant",
        "production",
        "semiconductor",
        "shipping",
        "supply chain",
        "warehouse",
    ],
}


@dataclass(frozen=True, slots=True)
class FeedSource:
    name: str
    url: str
    industry_focus: str


FEED_PROFILES = {
    "finance": (
        FeedSource(
            name="google-news-finance",
            url=(
                "https://news.google.com/rss/search?"
                "q=finance+OR+banking+OR+stocks+OR+economy&hl=en-US&gl=US&ceid=US:en"
            ),
            industry_focus="Finance",
        ),
    ),
    "manufacturing": (
        FeedSource(
            name="google-news-manufacturing",
            url=(
                "https://news.google.com/rss/search?"
                "q=manufacturing+OR+factory+OR+production+OR+%22supply+chain%22"
                "&hl=en-US&gl=US&ceid=US:en"
            ),
            industry_focus="Manufacturing",
        ),
    ),
}
FEED_PROFILES["mixed"] = FEED_PROFILES["finance"] + FEED_PROFILES["manufacturing"]


def list_feed_profiles() -> tuple[str, ...]:
    return tuple(FEED_PROFILES)


def _strip_html(value: str) -> str:
    text = unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return _strip_html(element.text)


def _parse_datetime(raw_value: str) -> str:
    if not raw_value:
        return ""

    try:
        parsed = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError, IndexError):
        return raw_value

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _score_profile_match(text: str, profile: str) -> int:
    lowered = text.lower()
    keywords = PROFILE_KEYWORDS.get(profile, [])
    return sum(1 for keyword in keywords if keyword in lowered)


def _infer_industry_focus(title: str, content: str) -> str:
    combined_text = " ".join([title, content])
    finance_score = _score_profile_match(combined_text, "finance")
    manufacturing_score = _score_profile_match(combined_text, "manufacturing")

    if manufacturing_score > finance_score and manufacturing_score > 0:
        return "Manufacturing"
    if finance_score > 0:
        return "Finance"
    return "General Business"


def _iter_rss_records(
    source: FeedSource,
    timeout_seconds: int = 10,
) -> Iterable[dict[str, str]]:
    request = Request(
        source.url,
        headers={"User-Agent": "Industry-Insight-Agent/1.0"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        xml_content = response.read()

    root = ET.fromstring(xml_content)
    for item in root.findall(".//item"):
        title = _safe_text(item.find("title"))
        description = _safe_text(item.find("description"))
        link = _safe_text(item.find("link"))
        published_at = _parse_datetime(_safe_text(item.find("pubDate")))

        if not title and not description:
            continue

        yield {
            "title": title,
            "content": description or title,
            "industry": source.industry_focus,
            "source_name": source.name,
            "source_url": link,
            "published_at": published_at,
            "collection_method": "live_rss",
            "feed_profile": source.industry_focus.lower(),
        }


def collect_rss_profile(
    profile: str = "mixed",
    max_articles: int = 100,
    timeout_seconds: int = 10,
) -> pd.DataFrame:
    if profile not in FEED_PROFILES:
        available = ", ".join(FEED_PROFILES)
        raise ValueError(f"Unknown feed profile '{profile}'. Available: {available}")

    records: list[dict[str, str]] = []
    for source in FEED_PROFILES[profile]:
        try:
            for record in _iter_rss_records(source, timeout_seconds=timeout_seconds):
                records.append(record)
                if len(records) >= max_articles:
                    return pd.DataFrame(records)
        except (HTTPError, URLError, TimeoutError, ET.ParseError, OSError):
            continue

    return pd.DataFrame(records)


def _normalize_local_dataset(dataset_name: str) -> pd.DataFrame:
    raw_df = load_dataset(dataset_name)

    if dataset_name == "financial":
        df = pd.DataFrame(
            {
                "title": raw_df["Title"].fillna("").astype(str),
                "content": raw_df["Content"].fillna("").astype(str),
                "industry": raw_df["Tag"].fillna("").astype(str),
            }
        )
    elif dataset_name == "google_daily_news":
        df = pd.DataFrame(
            {
                "title": raw_df["headline"].fillna("").astype(str),
                "content": raw_df["summary"].fillna("").astype(str),
                "industry": raw_df["category"].fillna("").astype(str),
                "source_name": raw_df["source"].fillna("").astype(str),
                "source_url": raw_df["url"].fillna("").astype(str),
                "published_at": raw_df["datetime"].fillna("").astype(str),
            }
        )
    else:
        raise ValueError(f"Local seed dataset '{dataset_name}' is not supported.")

    df["source_name"] = df.get("source_name", pd.Series("", index=df.index))
    df["source_url"] = df.get("source_url", pd.Series("", index=df.index))
    df["published_at"] = df.get("published_at", pd.Series("", index=df.index))
    df["collection_method"] = "local_seed"
    return df


def _filter_by_profile(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile == "mixed":
        return df

    combined_text = (
        df["title"].fillna("").astype(str) + " " + df["content"].fillna("").astype(str)
    )
    scores = combined_text.map(lambda value: _score_profile_match(value, profile))
    filtered_df = df[scores > 0]

    if filtered_df.empty:
        return df.head(0)
    return filtered_df


def build_local_seed_dataset(
    profile: str = "mixed",
    max_articles: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    frames = [
        _normalize_local_dataset("financial"),
        _normalize_local_dataset("google_daily_news"),
    ]
    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = _filter_by_profile(combined_df, profile=profile)
    if combined_df.empty:
        combined_df = pd.concat(frames, ignore_index=True)

    if profile == "mixed" and len(combined_df) > max_articles:
        finance_df = _filter_by_profile(combined_df, profile="finance")
        manufacturing_df = _filter_by_profile(combined_df, profile="manufacturing")
        finance_target = max_articles // 2
        manufacturing_target = max_articles - finance_target
        sampled_frames = []

        if not finance_df.empty:
            sampled_frames.append(
                finance_df.sample(
                    n=min(finance_target, len(finance_df)),
                    random_state=random_state,
                )
            )
        if not manufacturing_df.empty:
            sampled_frames.append(
                manufacturing_df.sample(
                    n=min(manufacturing_target, len(manufacturing_df)),
                    random_state=random_state,
                )
            )

        if sampled_frames:
            combined_df = pd.concat(sampled_frames, ignore_index=True).drop_duplicates()

    if len(combined_df) > max_articles:
        combined_df = combined_df.sample(
            n=max_articles,
            random_state=random_state,
        )

    combined_df["industry"] = combined_df.apply(
        lambda row: _infer_industry_focus(row["title"], row["content"]),
        axis=1,
    )
    combined_df["feed_profile"] = profile
    combined_df["collected_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def save_ingested_dataset(
    df: pd.DataFrame,
    output_path: Path = INGESTED_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def collect_news(
    profile: str = "mixed",
    max_articles: int = 100,
    *,
    prefer_live: bool = True,
    allow_fallback: bool = True,
    timeout_seconds: int = 10,
    output_path: Path = INGESTED_PATH,
) -> pd.DataFrame:
    if prefer_live:
        live_df = collect_rss_profile(
            profile=profile,
            max_articles=max_articles,
            timeout_seconds=timeout_seconds,
        )
        if not live_df.empty:
            live_df["collected_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            save_ingested_dataset(live_df, output_path=output_path)
            return live_df

    if not allow_fallback:
        raise RuntimeError("Live RSS collection returned no records and fallback is disabled.")

    fallback_df = build_local_seed_dataset(
        profile=profile,
        max_articles=max_articles,
    )
    save_ingested_dataset(fallback_df, output_path=output_path)
    return fallback_df
