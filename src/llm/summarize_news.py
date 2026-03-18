from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(slots=True)
class SummaryResult:
    summary: str
    method: str


@dataclass(slots=True)
class NewsSummarizer:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    force_fallback: bool = False
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if self.force_fallback or not self.api_key:
            return

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        except Exception:
            self._client = None

    def summarize(
        self, title: str, content: str, category: str = "", industry: str = ""
    ) -> SummaryResult:
        if self._client is not None:
            summary = self._summarize_with_openai(
                title=title,
                content=content,
                category=category,
                industry=industry,
            )
            if summary is not None:
                return SummaryResult(summary=summary, method="openai")

        summary = summarize_with_rules(
            title=title,
            content=content,
            category=category,
            industry=industry,
        )
        return SummaryResult(summary=summary, method="rule_based")

    def _summarize_with_openai(
        self, title: str, content: str, category: str = "", industry: str = ""
    ) -> str | None:
        if self._client is None:
            return None

        prompt = (
            "Summarize the following industry news article in 2-3 concise sentences. "
            "Focus on the core event, operational or business implication, and why it matters.\n\n"
            f"Category: {category or 'N/A'}\n"
            f"Industry hint: {industry or 'N/A'}\n"
            f"Title: {title}\n"
            f"Content: {content[:3000]}"
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize business news for analysts. "
                            "Keep summaries factual, concise, and decision-oriented."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return None

        raw_content = completion.choices[0].message.content or ""
        return clean_summary(raw_content)


def clean_summary(summary: str, max_length: int = 360) -> str:
    cleaned = re.sub(r"\s+", " ", summary).strip()
    if not cleaned:
        return ""
    return textwrap.shorten(cleaned, width=max_length, placeholder="...")


def summarize_with_rules(
    title: str, content: str, category: str = "", industry: str = ""
) -> str:
    cleaned_title = re.sub(r"\s+", " ", title).strip()
    cleaned_content = re.sub(r"\s+", " ", content).strip()

    if cleaned_title and cleaned_content.lower().startswith(cleaned_title.lower()):
        cleaned_content = cleaned_content[len(cleaned_title) :].strip(" .:-")

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_content)
    selected_sentences: list[str] = []

    for sentence in sentences:
        normalized = sentence.strip(" .")
        if len(normalized) < 35:
            continue
        if cleaned_title and normalized.lower() == cleaned_title.lower():
            continue
        selected_sentences.append(normalized)
        if len(selected_sentences) == 3:
            break

    if not selected_sentences:
        base = cleaned_title or cleaned_content or "No summary available."
        return clean_summary(base)

    summary_sentences: list[str] = []
    intro_parts = [part for part in [category, industry] if part]
    if intro_parts and cleaned_title:
        summary_sentences.append(f"{cleaned_title} [{', '.join(intro_parts)}].")
    elif cleaned_title:
        summary_sentences.append(f"{cleaned_title}.")

    for sentence in selected_sentences:
        if sentence not in summary_sentences:
            summary_sentences.append(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}.")

    return clean_summary(" ".join(summary_sentences))
