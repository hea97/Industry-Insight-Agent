from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


CATEGORY_KEYWORDS = {
    "Market": [
        "bond",
        "cpi",
        "economy",
        "futures",
        "index",
        "inflation",
        "market",
        "nasdaq",
        "rate",
        "shares",
        "stock",
        "trading",
        "unemployment",
        "yield",
    ],
    "Company": [
        "announced",
        "ceo",
        "earnings",
        "forecast",
        "guidance",
        "margin",
        "profit",
        "quarter",
        "quarterly",
        "results",
        "revenue",
        "sales",
    ],
    "Policy": [
        "antitrust",
        "approval",
        "central bank",
        "congress",
        "court",
        "fed",
        "government",
        "law",
        "policy",
        "regulation",
        "sanction",
        "tariff",
        "tax",
    ],
    "Technology": [
        "ai",
        "analytics",
        "automation",
        "chip",
        "cloud",
        "cybersecurity",
        "data center",
        "digital",
        "patent",
        "platform",
        "robot",
        "semiconductor",
        "software",
        "technology",
    ],
    "Supply Chain": [
        "delivery",
        "export",
        "freight",
        "import",
        "inventory",
        "logistics",
        "orders",
        "shipment",
        "shipping",
        "supplier",
        "supply chain",
        "warehouse",
    ],
    "Manufacturing": [
        "assembly",
        "capacity",
        "factory",
        "facility",
        "manufacturing",
        "operations",
        "output",
        "plant",
        "production",
        "rollout",
        "shutdown",
        "workforce",
    ],
    "Investment": [
        "acquisition",
        "buyback",
        "capital raise",
        "deal",
        "financing",
        "funding",
        "investment",
        "ipo",
        "merger",
        "partnership",
        "private equity",
        "stake",
    ],
    "Sustainability": [
        "battery",
        "carbon",
        "climate",
        "emission",
        "energy transition",
        "esg",
        "green",
        "net zero",
        "renewable",
        "solar",
        "sustainability",
    ],
}

DEFAULT_CATEGORY = "General"

CATEGORY_DESCRIPTIONS = {
    "Market": "Market movement, macro indicators, trading conditions, or investor reaction.",
    "Company": "Company-level updates such as earnings, guidance, performance, or management actions.",
    "Policy": "Government policy, regulation, interest rates, legal action, or public-sector intervention.",
    "Technology": "AI, software, chips, digital transformation, or technology-driven business change.",
    "Supply Chain": "Shipping, inventory, trade flows, sourcing, logistics, or supplier disruptions.",
    "Manufacturing": "Factory operations, production capacity, plant activity, or industrial execution.",
    "Investment": "M&A, capital allocation, fundraising, partnerships, or strategic investments.",
    "Sustainability": "Climate, emissions, ESG, energy transition, or sustainability initiatives.",
    "General": "Important business news that does not fit the categories above.",
}


@dataclass(slots=True)
class ClassificationResult:
    category: str
    method: str


@dataclass(slots=True)
class NewsClassifier:
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

    def classify(
        self, title: str, content: str, industry: str = ""
    ) -> ClassificationResult:
        if self._client is not None:
            category = self._classify_with_openai(
                title=title,
                content=content,
                industry=industry,
            )
            if category is not None:
                return ClassificationResult(category=category, method="openai")

        category = classify_with_rules(title=title, content=content, industry=industry)
        return ClassificationResult(category=category, method="rule_based")

    def _classify_with_openai(
        self, title: str, content: str, industry: str = ""
    ) -> str | None:
        if self._client is None:
            return None

        category_list = "\n".join(
            f"- {name}: {description}"
            for name, description in CATEGORY_DESCRIPTIONS.items()
        )
        prompt = (
            "Classify the following industry news article into exactly one category.\n"
            "Return only the category label.\n\n"
            f"Categories:\n{category_list}\n\n"
            f"Industry hint: {industry or 'N/A'}\n"
            f"Title: {title}\n"
            f"Content: {content[:3000]}"
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise business news classifier. "
                            "Answer with one category label only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return None

        raw_content = completion.choices[0].message.content or ""
        return normalize_category(raw_content)


def normalize_category(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        return None

    if cleaned in CATEGORY_DESCRIPTIONS:
        return cleaned

    lowered = cleaned.lower()
    for category in CATEGORY_DESCRIPTIONS:
        if lowered == category.lower():
            return category

    for category in CATEGORY_DESCRIPTIONS:
        if category.lower() in lowered:
            return category

    return None


def classify_with_rules(title: str, content: str, industry: str = "") -> str:
    combined_text = " ".join([title, industry, content]).lower()
    scores = {category: 0 for category in CATEGORY_KEYWORDS}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                scores[category] += 1

    industry_lower = industry.lower()
    if "finance" in industry_lower:
        scores["Market"] += 2
        scores["Company"] += 1
    if "manufacturing" in industry_lower:
        scores["Manufacturing"] += 2
        scores["Supply Chain"] += 1
    if "technology" in industry_lower:
        scores["Technology"] += 2
    if "supply chain" in industry_lower:
        scores["Supply Chain"] += 2
    if "policy" in industry_lower:
        scores["Policy"] += 2

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return DEFAULT_CATEGORY
    return best_category
