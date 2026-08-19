# CRO Research Intelligence Agent

Agent prototype for turning clinical research PDFs into a structured evidence table for CRO feasibility review.

The project started as a small experiment: could a folder of oncology and clinical trial papers be parsed, normalized, and summarized into fields that are actually useful for research operations? The current version uses GROBID for scientific PDF parsing, Python for TEI XML processing and output generation, and Groq/Llama 3.1 for structured extraction from the article text.

The agent is not meant to replace scientific or clinical review. The goal is to reduce the first-pass manual burden of reading papers, pulling trial details, and identifying articles that deserve closer human review.

**Live App:** [cro-research-intelligence-agent.streamlit.app](https://cro-research-intelligence-agent-cmb2aegsutbaqryqblggzb.streamlit.app/)

## What it does

Contract research teams often need to skim a stack of published trials to get a first read on operational feasibility - study design, population, endpoints, and anything that looks recruitment- or safety-heavy. Doing that by hand for more than a handful of papers is slow. The script runs that first pass and leaves a paper trail (`extraction_quality_score`, `review_priority`) so a human reviewer knows where to look closer.

The workflow supports two sources of articles:

- a static, manually collected PDF set
- open-access articles pulled in on demand from Europe PMC

At a high level:

1. Find PDFs in the local article corpus (recursively, across subfolders)
2. Pull in new open-access articles from Europe PMC first
3. Parse each PDF into TEI XML with GROBID
4. Extract title, abstract, and major sections from the XML
5. Send one consolidated payload per article to Groq for structured extraction
6. Score extraction quality and flag a review priority
7. Write CSV, Markdown, and PDF reports
8. Optionally browse the results in a small read-only Streamlit viewer

## Static article corpus

`CT_RCT_Studies/` holds the PDFs. It's split by where an article came from:

- `initial_pubmed_set/` - the original manually collected papers
- `fetched_europe_pmc/` - anything pulled in later by Open-Access Article Intake

PDFs sitting directly in `CT_RCT_Studies/` (not yet sorted into a subfolder) still get picked up and processed - they're just labeled `root_legacy` in the output so you know they haven't been organized yet. Every row in the evidence table carries a `corpus_source` value (`initial_pubmed_set`, `fetched_europe_pmc`, `root_legacy`, or `unknown`) so you can always tell where an article came from.

## Open-Access Article Intake

The static corpus is useful but it goes stale. This is the other half of the project: instead of manually downloading PDFs, the workflow can search for and pull in new articles through Europe PMC, then feed them straight into the same local pipeline. It's a structured API integration against Europe PMC's REST API. No general web scraping, and nothing paywalled ever gets touched.

Here's what actually happens when you run it. You give it a search query and a date window (`--query`, `--days-back`), and it searches Europe PMC's REST API restricted to `OPEN_ACCESS:y AND PUB_TYPE:"research-article"` - so it's not pulling in general search results, just open-access research articles from that window. Each hit comes back with real metadata (title, authors, journal, DOI, PMCID, PMID, publication date, open-access flag), which gets checked against what's already in `outputs/europe_pmc_intake.csv` and against filenames already sitting under `CT_RCT_Studies/`. Anything already known gets skipped.

For anything new, it looks for an open-access PDF link and downloads it into `CT_RCT_Studies/fetched_europe_pmc/` - but only after checking the response is actually a PDF, not just a 200 status. If there's no PDF but full-text XML is available, that gets saved to `outputs/source_xml/` instead. Either way, every candidate gets logged to `outputs/europe_pmc_intake.csv` with an `ingestion_status` (`downloaded_pdf`, `downloaded_xml_metadata_only`, `metadata_only`, `download_unavailable`, or `skipped_duplicate`) - nothing gets marked downloaded unless it actually downloaded. Once intake wraps up, the run continues straight into the normal GROBID + Groq pipeline; fetched articles don't get any special-cased treatment once they're sitting on disk.

By default `--fetch-new` only parses whatever it just fetched, not the whole corpus - so adding a handful of new articles doesn't mean re-parsing everything you already have. Add `--full-rebuild` if you actually want that (rebuilding the compiled table from every PDF in `CT_RCT_Studies/`, which is also just what happens if you run the script without `--fetch-new` at all).

```bash
python cro_research_intelligence_agent.py \
  --fetch-new \
  --query "oncology randomized controlled trial immunotherapy" \
  --days-back 30 \
  --max-new-articles 5
```

If a search comes back with nothing new, it logs that and moves on into the normal pipeline instead of failing.

**Query modes.** `--query` above is manual mode, and it's still the default - nothing changes if you don't pass `--query-mode`. There's also an auto mode that draws from a curated, reproducible pool of full search phrases spanning several biomedical domains (oncology, rare disease, infectious disease, neurology, autoimmune disease, liver/metabolic disease, digital health, patient-reported outcomes, and feasibility/operations-oriented studies) instead of searching the same fixed phrase every time:

```bash
python cro_research_intelligence_agent.py \
  --fetch-new \
  --query-mode auto \
  --query-count 3 \
  --query-seed 42 \
  --days-back 90 \
  --max-new-articles 10
```

The same `--query-seed` always generates the same query list. `--max-new-articles` is a total cap across all generated queries in a run, not a per-query limit. Each candidate's manifest row still records the exact query that found it. Add `--dry-run-intake` to print the queries a run would use without downloading anything or writing to `outputs/europe_pmc_intake.csv`.

## Corpus structure

```
CT_RCT_Studies/
├── initial_pubmed_set/       # original manually collected PDFs
├── fetched_europe_pmc/       # articles added by Open-Access Article Intake
└── *.pdf                     # anything not yet sorted ("root_legacy")

data/
└── compiled_article_evidence_table.csv  # the committed processed evidence table - the app's data source

outputs/
├── grobid_xml/                    # cached GROBID XML, one file per article
├── llm_json/                      # cached Groq extraction JSON, content-hash keyed
├── source_xml/                    # full-text XML from intake, when no PDF was available
├── europe_pmc_intake.csv          # intake/provenance log - what Open-Access Article Intake found, downloaded, or skipped
├── compiled_article_evidence_summary.md
└── compiled_article_evidence_report.pdf
```

`data/compiled_article_evidence_table.csv` is the single committed evidence table - both the CLI pipeline and the Streamlit app read/write this same file, so there's no separate sample or deployment copy to keep in sync. Everything else under `outputs/` is a generated report, cache, or local runtime artifact and isn't committed.

If you're starting from an older copy of this project where the PDFs are all sitting loose in `CT_RCT_Studies/`, this moves them into the initial set without touching anything else (GROBID/LLM caches are keyed by filename, not path, so this is safe):

```bash
mkdir -p "CT_RCT_Studies/initial_pubmed_set"
find "CT_RCT_Studies" -maxdepth 1 -name "*.pdf" -exec mv {} "CT_RCT_Studies/initial_pubmed_set/" \;
```

## GROBID and Groq LLM

GROBID does the PDF parsing - it converts each PDF into TEI XML, which is a much more tractable format than raw PDF text for pulling out title, authors, abstract, and body sections. The XML gets cached per article so re-running the pipeline doesn't re-upload PDFs that were already processed.

Only the extracted text - never the raw PDF - goes to Groq. For each article, the script sends one consolidated prompt with the title and whatever major sections were extracted. Groq returns structured JSON for study type, phase, PICO fields, endpoints, biomarkers, and CRO feasibility fields such as recruitment complexity, endpoint burden, safety monitoring needs, and overall risk. Fields that aren't clearly stated come back as `not_reported` or `not_applicable` rather than guessed at.


## Outputs

- **`data/compiled_article_evidence_table.csv`** - the committed evidence dataset used by the Streamlit viewer (the only file under `data/`, everything else below lives in `outputs/`). One row per article that actually went through GROBID + Groq (both the manually collected corpus and anything fetched via intake), led by reader-facing fields (title, journal, study type, feasibility risk, plain-English blurb), then study design, PICO, endpoints, biomarkers, feasibility fields, and diagnostics (`corpus_source`, `doi`, `title_quality_flag`, `abstract_source`, `extraction_quality_flag`, `missing_field_count`)
- **`compiled_article_evidence_summary.md`** - a readable per-article writeup with a plain-English summary and the feasibility assessment
- **`compiled_article_evidence_report.pdf`** - the same summary exported as a PDF with ReportLab
- **`europe_pmc_intake.csv`** - intake/provenance log, only populated when `--fetch-new` is used. Tracks every candidate Europe PMC returned - downloaded, skipped, duplicate, or metadata-only - regardless of whether it was ever parsed. Not merged into the evidence table.

`extraction_quality_score` and `review_priority` exist so you don't have to eyeball 25 rows to find the ones worth checking by hand - low scores or failed extractions get flagged `high` priority automatically.

## Streamlit


```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```

## Setup

Requirements: Python, Docker (for GROBID), Groq key.

```bash
pip install requests python-dotenv groq reportlab
```

```bash
docker pull lfoppiano/grobid:0.8.0
docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

```bash
cp .env.example .env
# edit .env and add your GROQ_API_KEY
```

## Commands

Run the full local corpus:
```bash
python cro_research_intelligence_agent.py
```

Fetch new open-access articles and parse just those (incremental, the default with `--fetch-new`):
```bash
python cro_research_intelligence_agent.py --fetch-new --query "oncology randomized controlled trial immunotherapy" --days-back 30 --max-new-articles 5
```

Same as above, but reparse the whole corpus afterward instead of just what was fetched:
```bash
python cro_research_intelligence_agent.py --fetch-new --full-rebuild --query "oncology randomized controlled trial immunotherapy"
```

Browse results:
```bash
streamlit run streamlit_app.py
```

## Limitations and Disclaimers

This is a prototype, not a clinical or regulatory tool.

- **Human review required.** LLM extraction can misread or oversimplify study details. Nothing here should be treated as validated without someone checking the source article.
- **No clinical recommendations.** Feasibility scores and risk ratings are planning aids for a CRO analyst, not clinical or regulatory judgments.
- **No paywalled or scraped content.** Open-Access Article Intake only uses the Europe PMC REST API and only downloads content explicitly marked open-access.
- Requires GROBID running locally.
- Extraction quality depends heavily on how cleanly GROBID parses a given PDF's structure - a few source PDFs (particularly some NEJM and conference-proceeding layouts) still come through with thin methods/results sections despite the positional section tracking.
- No validation against ground truth data - extraction accuracy hasn't been formally measured against a labeled sample.
- Single-threaded, processes one article at a time.
