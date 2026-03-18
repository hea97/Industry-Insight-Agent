# Industry Insight Agent

Automation-first system for turning unstructured industry news into structured insight.

`Collection -> Cleaning -> LLM Classification -> LLM Summary -> Structured Storage -> Trend Analysis -> Dashboard`

## Overview

This project focuses on a real automation workflow rather than a one-off LLM demo.

- Collect industry news from automated sources or saved CSV datasets
- Normalize messy, unstructured text into a consistent schema
- Classify each article with an LLM-oriented analysis engine
- Generate concise summaries for faster review
- Store results as analysis-ready structured data
- Visualize category, industry, and keyword trends in Streamlit

## Why It Matters

Industry analysts and operations teams deal with information overload every day.
Manual review is slow, repetitive, and difficult to scale. This project shows how
LLM capabilities can be embedded into a workflow that supports business monitoring
and decision-making.

## Architecture

```text
News Feeds / Raw CSV
        |
        v
Automated Collection Layer
        |
        v
Data Cleaning / Normalization
        |
        v
LLM Analysis Engine
   |- Classification
   |- Summarization
        |
        v
Structured Result CSV
        |
        v
Trend Analysis + Streamlit Dashboard
```

## Project Structure

```text
Industry-Insight-Agent/
- app.py
- data/
- requirements.txt
- src/
  - data/
    - collect_news.py
    - load_data.py
  - llm/
    - classify_news.py
    - summarize_news.py
  - pipeline.py
```

## Key Features

- Two input modes: saved CSV datasets or automated ingestion
- Live RSS-first ingestion with local-seed fallback for resilient demos
- API-first LLM flow with rule-based fallback when no API key is available
- Structured output with source, collection, category, and summary metadata
- Portfolio-friendly dashboard for trend analysis and workflow explanation

## Installation

```bash
pip install -r requirements.txt
```

Optional environment variables:

```bash
set OPENAI_API_KEY=your_key
set OPENAI_MODEL=gpt-4o-mini
```

## Run The Pipeline

Analyze a saved CSV dataset:

```bash
python -m src.pipeline --source-mode dataset --dataset financial --max-rows 100 --sampling sample
```

Run the automated ingestion pipeline using local seed data only:

```bash
python -m src.pipeline --source-mode ingest --profile mixed --max-rows 100 --local-seed-only --fallback-only
```

Run the automated ingestion pipeline with live RSS collection enabled:

```bash
python -m src.pipeline --source-mode ingest --profile manufacturing --max-rows 100
```

## Run The Dashboard

```bash
streamlit run app.py
```

The dashboard exposes:

- input mode selection
- automated ingestion profile selection
- classification and summarization status
- category distribution
- collection breakdown
- industry versus category comparison
- keyword analysis

## Output Schema

The result dataset can include:

- `record_id`
- `source_dataset`
- `source_mode`
- `industry_profile`
- `title`
- `content`
- `industry`
- `category`
- `summary`
- `source_name`
- `source_url`
- `collection_method`
- `published_at`
- `collected_at`
- `classification_method`
- `summary_method`
- `processed_at`

## Portfolio Positioning

This repository is strongest when presented as:

- an LLM-enabled automation system
- a workflow for handling unstructured business data
- a small but complete decision-support pipeline

If you want the Hampking-focused version, see `포트폴리오.md`.
