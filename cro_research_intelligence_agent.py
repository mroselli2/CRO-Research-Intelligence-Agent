"""
CRO Research Intelligence Agent
Parses clinical research PDFs with GROBID, extracts structured CRO feasibility
intelligence with Groq, and writes an evidence table + summary report.
"""

import os
import re
import json
import csv
import time
import random
import shutil
import hashlib
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple, Callable

import requests
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

BASE_DIR = Path(__file__).resolve().parent
ARTICLES_DIR = BASE_DIR / "CT_RCT_Studies"

# Loose PDFs dropped directly in ARTICLES_DIR still get picked up, see infer_corpus_source().
INITIAL_PUBMED_DIR = ARTICLES_DIR / "initial_pubmed_set"
FETCHED_EUROPE_PMC_DIR = ARTICLES_DIR / "fetched_europe_pmc"

OUTPUT_DIR = BASE_DIR / "outputs"
GROBID_XML_DIR = OUTPUT_DIR / "grobid_xml"
LLM_JSON_DIR = OUTPUT_DIR / "llm_json"
SOURCE_XML_DIR = OUTPUT_DIR / "source_xml"
CROSSREF_CACHE_DIR = OUTPUT_DIR / "crossref_cache"
INTAKE_MANIFEST_PATH = OUTPUT_DIR / "europe_pmc_intake.csv"
LEGACY_MANIFEST_PATH = OUTPUT_DIR / "source_manifest.csv"

DATA_DIR = BASE_DIR / "data"
EVIDENCE_TABLE_PATH = DATA_DIR / "compiled_article_evidence_table.csv"
METADATA_OVERRIDES_PATH = BASE_DIR / "metadata" / "article_metadata_overrides.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
GROBID_XML_DIR.mkdir(exist_ok=True)
LLM_JSON_DIR.mkdir(exist_ok=True)
SOURCE_XML_DIR.mkdir(exist_ok=True)
CROSSREF_CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
INITIAL_PUBMED_DIR.mkdir(exist_ok=True)
FETCHED_EUROPE_PMC_DIR.mkdir(exist_ok=True)


def migrate_legacy_manifest() -> None:
    """One-time copy so a pre-rename outputs/source_manifest.csv isn't silently orphaned."""
    if not INTAKE_MANIFEST_PATH.exists() and LEGACY_MANIFEST_PATH.exists():
        shutil.copy2(LEGACY_MANIFEST_PATH, INTAKE_MANIFEST_PATH)
        print(f"Migrated {LEGACY_MANIFEST_PATH.name} -> {INTAKE_MANIFEST_PATH.name}")


migrate_legacy_manifest()

GROBID_BASE_URL = "http://localhost:8070"
GROBID_PROCESS_URL = f"{GROBID_BASE_URL}/api/processFulltextDocument"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please ensure .env file exists with your API key. "
        "See .env.example for template."
    )

client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# None = process everything found. Set --max-articles at the CLI for a quick test run.
MAX_ARTICLES = None

GROQ_RETRY_LIMIT = 3
GROQ_RETRY_WAIT_SECS = 10
INTER_ARTICLE_DELAY_SECS = 2

MAX_ABSTRACT_CHARS = 1500
MAX_METHODS_CHARS = 5000
MAX_RESULTS_CHARS = 5000
MAX_DISCUSSION_CHARS = 3000

# Rough proxy for how much the LLM actually had to work with - feeds review_priority.
CORE_INTELLIGENCE_FIELDS = [
    "disease_area", "study_type", "study_phase", "sample_size",
    "population", "intervention", "primary_endpoint", "operational_feasibility_risk",
]
NOT_REPORTED_VALUES = {"not_reported", "not_applicable", "", "unclear", "n/a"}


@dataclass
class ArticleData:
    article_id: str
    file_name: str
    corpus_source: str = "unknown"

    title: str = ""
    title_quality_flag: str = "unknown"
    authors: str = ""
    journal: str = ""
    publication_year: str = ""
    doi: str = "not_reported"

    abstract: str = ""
    abstract_source: str = "missing"
    methods_text: str = ""
    results_text: str = ""
    discussion_text: str = ""
    body_text_summary_source: str = ""
    references_count: int = 0
    extraction_quality_flag: str = "unknown"

    intelligence: Dict[str, Any] = field(default_factory=dict)

    abstract_char_count: int = 0
    methods_char_count: int = 0
    results_char_count: int = 0
    discussion_char_count: int = 0
    groq_payload_char_count: int = 0
    sections_sent_to_groq: str = ""
    llm_extraction_status: str = "pending"
    json_parse_error: str = ""
    source_excerpt: str = ""

    extraction_quality_score: int = 0
    missing_field_count: int = 0
    review_priority: str = "unknown"

    def intel(self, key: str, default: Any = "not_reported") -> Any:
        return self.intelligence.get(key, default)

    def to_csv_row(self) -> Dict[str, Any]:
        return {name: getter(self) for name, getter in FIELD_SPEC}


# Single source of truth for the CSV: column order here is column order in the file.
FIELD_SPEC: List[Tuple[str, Callable[[ArticleData], Any]]] = [
    # Reader-facing front - scannable at a glance.
    ("title", lambda a: a.title),
    ("journal", lambda a: a.journal),
    ("publication_year", lambda a: a.publication_year),
    ("study_type", lambda a: a.intel("study_type")),
    ("disease_area", lambda a: a.intel("disease_area")),
    ("therapeutic_area", lambda a: a.intel("therapeutic_area")),
    ("operational_feasibility_risk", lambda a: a.intel("operational_feasibility_risk")),
    ("review_priority", lambda a: a.review_priority),
    ("extraction_quality_score", lambda a: a.extraction_quality_score),
    # Study design, PICO, endpoints, feasibility detail.
    ("study_phase", lambda a: a.intel("study_phase")),
    ("sample_size", lambda a: a.intel("sample_size")),
    ("randomization", lambda a: a.intel("randomization")),
    ("intervention_type", lambda a: a.intel("intervention")),  # LLM schema key is "intervention"
    ("comparator", lambda a: a.intel("comparator")),
    ("follow_up_period", lambda a: a.intel("follow_up_period")),
    ("population", lambda a: a.intel("population")),
    ("population_summary", lambda a: str(a.intel("population", ""))[:200]),
    ("primary_endpoint", lambda a: a.intel("primary_endpoint")),
    ("secondary_endpoints", lambda a: str(a.intel("secondary_endpoints", []))),
    ("exploratory_endpoints", lambda a: str(a.intel("exploratory_endpoints", []))),
    ("biomarkers", lambda a: str(a.intel("biomarkers", []))),
    ("genetic_or_molecular_markers", lambda a: str(a.intel("genetic_or_molecular_markers", []))),
    ("inclusion_criteria", lambda a: str(a.intel("inclusion_criteria", []))),
    ("exclusion_criteria", lambda a: str(a.intel("exclusion_criteria", []))),
    ("reported_adverse_events", lambda a: str(a.intel("reported_adverse_events", []))),
    ("safety_signals", lambda a: str(a.intel("safety_signals", []))),
    ("recruitment_complexity", lambda a: a.intel("recruitment_complexity")),
    ("eligibility_restrictions", lambda a: a.intel("eligibility_restrictions")),
    ("biomarker_testing_requirements", lambda a: a.intel("biomarker_testing_requirements")),
    ("endpoint_burden", lambda a: a.intel("endpoint_burden")),
    ("visit_schedule_demands", lambda a: a.intel("visit_schedule_demands")),
    ("safety_monitoring_needs", lambda a: a.intel("safety_monitoring_needs")),
    ("site_activation_considerations", lambda a: a.intel("site_activation_considerations")),
    ("feasibility_risk_reason", lambda a: a.intel("feasibility_risk_reason", "")),
    ("feasibility_summary", lambda a: a.intel("feasibility_summary", "")),
    ("plain_english_blurb", lambda a: a.intel("plain_english_blurb", "")),
    # Provenance, metadata, and diagnostics.
    ("corpus_source", lambda a: a.corpus_source),
    ("authors", lambda a: a.authors),
    ("article_id", lambda a: a.article_id),
    ("file_name", lambda a: a.file_name),
    ("doi", lambda a: a.doi),
    ("title_quality_flag", lambda a: a.title_quality_flag),
    ("extraction_quality_flag", lambda a: a.extraction_quality_flag),
    ("missing_field_count", lambda a: a.missing_field_count),
    ("abstract", lambda a: a.abstract),
    ("abstract_source", lambda a: a.abstract_source),
    ("abstract_char_count", lambda a: a.abstract_char_count),
    ("methods_char_count", lambda a: a.methods_char_count),
    ("results_char_count", lambda a: a.results_char_count),
    ("discussion_char_count", lambda a: a.discussion_char_count),
    ("groq_payload_char_count", lambda a: a.groq_payload_char_count),
    ("sections_sent_to_groq", lambda a: a.sections_sent_to_groq),
    ("llm_extraction_status", lambda a: a.llm_extraction_status),
    ("json_parse_error", lambda a: a.json_parse_error),
    ("source_excerpt", lambda a: a.source_excerpt),
]

CSV_HEADERS = [name for name, _ in FIELD_SPEC]


def send_pdf_to_grobid(pdf_path: Path) -> str:
    print(f"  Sending {pdf_path.name} to GROBID...")
    with open(pdf_path, "rb") as pdf_file:
        response = requests.post(GROBID_PROCESS_URL, files={"input": pdf_file}, timeout=120)
        response.raise_for_status()
    return response.text


def save_grobid_xml(article_id: str, xml_content: str) -> Path:
    xml_path = GROBID_XML_DIR / f"{article_id}.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return xml_path


def load_grobid_xml(article_id: str) -> Optional[str]:
    xml_path = GROBID_XML_DIR / f"{article_id}.xml"
    return xml_path.read_text(encoding="utf-8") if xml_path.exists() else None


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) > max_chars:
        return text[:max_chars] + "...", True
    return text, False


def clean_text(text: str) -> str:
    """Collapse whitespace left over from joining itertext() fragments."""
    return " ".join(text.split())


def paragraph_text(p: ET.Element) -> str:
    """Full text of a paragraph, including anything after inline tags like <ref> or <hi>
    that reading p.text alone would silently drop."""
    return clean_text("".join(p.itertext()))


SECTION_KEYWORDS = {
    "methods": ["method", "materials", "patient", "design", "trial", "participant", "analysis"],
    "results": ["result", "finding", "outcome", "efficacy", "safety", "effect"],
    "discussion": ["discussion", "interpretation", "limitation", "conclusion", "implication"],
}

# Ends a section without starting a new one, otherwise trailing boilerplate (funding,
# conflicts of interest, etc.) gets attributed to whatever section came before it.
SECTION_BOUNDARY_KEYWORDS = [
    "reference", "acknowledg", "declaration", "author contribution",
    "availability", "supplement", "conflict of interest", "funding",
]


def classify_heading(heading: str) -> Optional[str]:
    heading = heading.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in heading for kw in keywords):
            return section
    return None


def is_section_boundary(heading: str) -> bool:
    heading = heading.lower()
    return any(kw in heading for kw in SECTION_BOUNDARY_KEYWORDS)


def looks_like_section_banner(heading: str, paragraph_count: int) -> bool:
    """Distinguishes a real top-level section banner (RESULTS, STAR METHODS) from a
    descriptive subsection heading that happens to contain a matching keyword (e.g. "Study
    design and vaccine generation" under a RESULTS banner). GROBID banners are consistently
    short, often all-caps, and usually carry no paragraph text of their own - subsection
    headings are longer, sentence-cased, and sit directly above real body text."""
    word_count = len(heading.split())
    return word_count <= 4 and (heading.isupper() or paragraph_count == 0)


def looks_like_citation_fragment(text: str) -> bool:
    """Flags GROBID title mis-extractions where a running header or reference-list
    fragment ends up in the title field instead of the real article title."""
    stripped = text.strip()
    if re.match(r"^\d+\s*[•·|]", stripped):
        return True
    if re.search(r"\b\d{4}\s*[:;]\s*\d+\b", stripped):
        return True
    if re.search(r"\(\d{1,2}\s+[A-Za-z]+\)", stripped):
        return True
    if re.match(r"^(Department|Division|Institute|Institution|Centre|Center|Faculty|School|University|Hospital)\s+of\b",
                stripped, re.IGNORECASE):
        return True
    if stripped.count("(") > stripped.count(")"):
        return True
    words = stripped.split()
    if words and len(words) <= 8 and ":" not in stripped:
        if sum(1 for w in words[1:] if w[:1].isupper()) == 0:
            return True
    return False


def parse_tei_xml(xml_content: str, filename: str = "") -> Dict[str, Any]:
    """Parse GROBID's TEI XML into title/authors/abstract/section text, with fallbacks
    for a missing title and a missing abstract."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"    ERROR parsing XML: {e}")
        return {
            "title": "", "title_quality": "missing", "authors": "", "journal": "",
            "publication_year": "", "doi": "", "abstract": "", "abstract_source": "missing",
            "methods_text": "", "results_text": "", "discussion_text": "",
            "body_text_summary_source": "", "references_count": 0,
        }

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title = ""
    title_quality = "good"
    title_elem = root.find(".//tei:title[@type='main']", ns)
    if title_elem is not None and title_elem.text:
        title = title_elem.text.strip()

    if title and looks_like_citation_fragment(title):
        title = ""
        title_quality = "suspicious"

    if not title or len(title) < 10:
        if filename:
            fallback_title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
            if fallback_title:
                title = fallback_title[:200]
                title_quality = "fallback"
        if not title:
            title_quality = "missing"

    authors_list = []
    for author in root.findall(".//tei:author", ns):
        pers_name = author.find("tei:persName", ns)
        if pers_name is not None:
            name_parts = [elem.text.strip() for elem in pers_name if elem.text]
            if name_parts:
                authors_list.append(" ".join(name_parts))
    authors = "; ".join(authors_list)

    # Scoped to the article's own sourceDesc/biblStruct so a reference-list entry's
    # journal/DOI can't be picked up instead of the article's.
    journal = ""
    journal_elem = root.find(".//tei:sourceDesc//tei:biblStruct//tei:monogr/tei:title[@level='j']", ns)
    if journal_elem is not None and journal_elem.text:
        journal = journal_elem.text.strip()

    doi = ""
    doi_elem = root.find(".//tei:sourceDesc//tei:biblStruct//tei:idno[@type='DOI']", ns)
    if doi_elem is not None and doi_elem.text:
        doi = doi_elem.text.strip()

    publication_year = ""
    year_elem = root.find(".//tei:monogr/tei:imprint/tei:date", ns)
    if year_elem is not None:
        publication_year = year_elem.get("when", "").split("-")[0]

    # GROBID nests abstract text as <abstract><div><p>, not directly under <abstract>.
    abstract_paragraphs = [t for t in (paragraph_text(p) for p in root.findall(".//tei:abstract//tei:p", ns)) if t]
    abstract = " ".join(abstract_paragraphs)
    abstract_source = "grobid_abstract" if abstract else "missing"

    # Section banners and their content can be separated by several unlabeled subsection
    # divs, so track the active section positionally rather than per-div.
    section_text = {"methods": [], "results": [], "discussion": []}
    all_body_parts = []
    current_section = None

    for div in root.findall(".//tei:body/tei:div", ns):
        head = div.find("tei:head", ns)
        heading = (head.text or "").strip() if head is not None else ""
        paragraphs = [t for t in (paragraph_text(p) for p in div.findall(".//tei:p", ns)) if t]

        if heading and looks_like_section_banner(heading, len(paragraphs)):
            if is_section_boundary(heading):
                current_section = None
            else:
                matched = classify_heading(heading)
                if matched:
                    current_section = matched

        div_text = " ".join(paragraphs)
        if div_text:
            all_body_parts.append(div_text)
            if current_section:
                section_text[current_section].append(div_text)

    methods_text = clean_text(" ".join(section_text["methods"]))
    results_text = clean_text(" ".join(section_text["results"]))
    discussion_text = clean_text(" ".join(section_text["discussion"]))
    all_body_text = clean_text(" ".join(all_body_parts))

    # Fallback payload so an unrecognized document structure still sends the LLM something.
    body_text_summary_source = all_body_text[:2000] if all_body_text else ""

    if not abstract and all_body_text:
        if methods_text or results_text:
            fallback = clean_text(methods_text[:500] + " " + results_text[:500])
        else:
            fallback = all_body_text
        abstract = fallback[:1500]
        abstract_source = "body_fallback"

    references_count = len(root.findall(".//tei:biblStruct", ns))

    return {
        "title": title.strip(),
        "title_quality": title_quality,
        "authors": authors.strip(),
        "journal": journal.strip(),
        "publication_year": publication_year.strip(),
        "doi": doi.strip(),
        "abstract": abstract.strip(),
        "abstract_source": abstract_source,
        "methods_text": methods_text,
        "results_text": results_text,
        "discussion_text": discussion_text,
        "body_text_summary_source": body_text_summary_source,
        "references_count": references_count,
    }


def assess_extraction_quality(parsed: Dict[str, Any]) -> str:
    if parsed.get("title") and (parsed.get("abstract") or parsed.get("body_text_summary_source")):
        return "good"

    issues = []
    if not parsed.get("title"):
        issues.append("missing_title")
    if not parsed.get("abstract"):
        issues.append("missing_abstract")
    if not parsed.get("methods_text"):
        issues.append("missing_methods")
    if not parsed.get("results_text"):
        issues.append("missing_results")
    return ";".join(issues) if issues else "good"


def infer_corpus_source(pdf_path: Path) -> str:
    """Where a PDF came from, inferred from its location under CT_RCT_Studies/."""
    try:
        relative_parts = pdf_path.resolve().relative_to(ARTICLES_DIR.resolve()).parts
    except ValueError:
        return "unknown"
    if len(relative_parts) == 1:
        return "root_legacy"  # sitting directly in CT_RCT_Studies/, not yet sorted into a subfolder
    top_folder = relative_parts[0]
    if top_folder == INITIAL_PUBMED_DIR.name:
        return "initial_pubmed_set"
    if top_folder == FETCHED_EUROPE_PMC_DIR.name:
        return "fetched_europe_pmc"
    return "unknown"


CORPUS_SOURCE_ORDER = {"initial_pubmed_set": 0, "fetched_europe_pmc": 1, "root_legacy": 2, "unknown": 3}


def corpus_sort_key(pdf_path: Path) -> Tuple[int, str]:
    """Deliberate processing order: initial set, then fetched articles, then unsorted legacy PDFs."""
    return (CORPUS_SOURCE_ORDER[infer_corpus_source(pdf_path)], pdf_path.name.lower())


def discover_pdfs() -> List[Path]:
    return sorted(ARTICLES_DIR.rglob("*.pdf"), key=corpus_sort_key)


CROSSREF_WORKS_URL = "https://api.crossref.org/works"


def crossref_cache_path(doi: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", doi)
    return CROSSREF_CACHE_DIR / f"{safe}.json"


def lookup_journal_via_crossref(doi: str) -> str:
    """Journal/container title by DOI via CrossRef's free public API - no key required.
    Cached to disk so repeat runs never re-hit the network for a DOI already looked up."""
    cache_file = crossref_cache_path(doi)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")).get("journal", "")
        except (json.JSONDecodeError, OSError):
            pass

    journal = ""
    try:
        response = requests.get(
            f"{CROSSREF_WORKS_URL}/{doi}",
            headers={"User-Agent": "cro-research-intelligence-agent/1.0 (structured DOI metadata lookup)"},
            timeout=15,
        )
        if response.status_code == 200:
            titles = response.json().get("message", {}).get("container-title", [])
            journal = titles[0].strip() if titles else ""
    except requests.RequestException as e:
        print(f"  CrossRef lookup failed for {doi}: {e}")

    cache_file.write_text(json.dumps({"doi": doi, "journal": journal}), encoding="utf-8")
    return journal


def document_parsing_agent(pdf_path: Path, article_id: str) -> ArticleData:
    """Send a PDF through GROBID (or reuse cached XML) and parse it into an ArticleData record."""
    print(f"\n[{article_id}] Parsing document")

    article = ArticleData(article_id, pdf_path.name, infer_corpus_source(pdf_path))

    xml_content = load_grobid_xml(article_id)
    if xml_content:
        print("  Using cached GROBID XML")
    else:
        xml_content = send_pdf_to_grobid(pdf_path)
        save_grobid_xml(article_id, xml_content)
        print("  GROBID parsing complete")

    parsed = parse_tei_xml(xml_content, pdf_path.name)

    article.title = parsed["title"]
    article.title_quality_flag = parsed["title_quality"]
    article.authors = parsed["authors"]
    article.journal = parsed["journal"]
    article.publication_year = parsed["publication_year"]
    article.abstract = parsed["abstract"]
    article.abstract_source = parsed["abstract_source"]
    article.methods_text = parsed["methods_text"]
    article.results_text = parsed["results_text"]
    article.discussion_text = parsed["discussion_text"]
    article.body_text_summary_source = parsed["body_text_summary_source"]
    article.references_count = parsed["references_count"]

    manifest_rows = load_manifest_rows()
    manifest_match = match_manifest_row(article, parsed["doi"], manifest_rows)

    if article.title_quality_flag in ("fallback", "missing") and manifest_match and manifest_match.get("title"):
        article.title = manifest_match["title"].strip()
        article.title_quality_flag = "fallback"

    if parsed["doi"]:
        article.doi = parsed["doi"]
    elif manifest_match and manifest_match.get("doi"):
        article.doi = manifest_match["doi"].strip()
    else:
        article.doi = "not_reported"

    if not article.journal and manifest_match and manifest_match.get("journal"):
        article.journal = manifest_match["journal"].strip()
        print(f"  Journal (from intake manifest): {article.journal}")

    if not article.journal and article.doi != "not_reported":
        crossref_journal = lookup_journal_via_crossref(article.doi)
        if crossref_journal:
            article.journal = crossref_journal
            print(f"  Journal (from CrossRef): {crossref_journal}")

    article.abstract_char_count = len(article.abstract)
    article.methods_char_count = len(article.methods_text)
    article.results_char_count = len(article.results_text)
    article.discussion_char_count = len(article.discussion_text)

    article.extraction_quality_flag = assess_extraction_quality(parsed)

    print(f"  Parsed title: {article.title[:80]}")
    print(f"  Abstract: {article.abstract_char_count}ch ({article.abstract_source}) | "
          f"Methods: {article.methods_char_count}ch | Results: {article.results_char_count}ch")
    print(f"  Extraction quality: {article.extraction_quality_flag}")

    return article


INTELLIGENCE_SYSTEM_PROMPT = """You are a CRO (Contract Research Organization) research intelligence analyst.
Extract and classify research article information for clinical trial feasibility planning.

CLASSIFICATION RULES:
- disease_area: Extract from title/abstract. Use: oncology, cardiology, neurology, immunology, infectious_disease, etc. If unclear, use "unclear"
- therapeutic_area: Same as disease_area typically
- study_type: Classify as: clinical_trial, randomized_controlled_trial, observational_study, review, preclinical_study, case_report, editorial, or unclear
- study_phase: For clinical trials use I, II, II/III, III, IV, or not_applicable for non-trials
- sample_size: Only report if clearly stated as a number (e.g., "n=150"), otherwise "not_reported"
- randomization: Only report if explicitly mentioned (Yes, No, not_applicable for non-trials)
- population: Concise patient population description
- intervention: Treatment or study intervention
- comparator: Control or comparator group (if applicable)
- outcomes: Primary outcomes as stated
- primary_endpoint: Single primary endpoint if stated
- secondary_endpoints: List any secondary endpoints
- biomarkers: Only list if clearly mentioned in text
- genetic_or_molecular_markers: Only if mentioned
- inclusion_criteria: List if provided
- exclusion_criteria: List if provided
- reported_adverse_events: Only events explicitly reported
- safety_signals: Only if safety concerns are discussed
- feasibility_risk: low/moderate/high based on recruitment/eligibility complexity
- recruitment_complexity: Assessment of recruitment difficulty
- endpoint_burden: Assessment of endpoint assessment burden
- visit_schedule_demands: Assessment of visit frequency burden
- safety_monitoring_needs: Assessment of safety monitoring requirements
- operational_feasibility_risk: low/moderate/high for CRO planning
- feasibility_risk_reason: Brief explanation of risk drivers

IMPORTANT:
- Do NOT invent values. Use "not_reported" if not clearly stated.
- Use "not_applicable" for fields that don't apply to the study type.
- Distinguish between actual findings vs. not reported.
- Inference is OK for broad classifications (disease area, study type) from title/abstract.
- Be specific only when evidence is clear in provided text.

Return valid JSON only. No additional text."""

INTELLIGENCE_JSON_SCHEMA = """{
  "plain_english_blurb": "2-3 sentence summary for non-specialists",
  "disease_area": "",
  "therapeutic_area": "",
  "study_type": "",
  "study_phase": "",
  "sample_size": "",
  "randomization": "",
  "population": "",
  "intervention": "",
  "comparator": "",
  "outcomes": "",
  "primary_endpoint": "",
  "secondary_endpoints": [],
  "biomarkers": [],
  "genetic_or_molecular_markers": [],
  "inclusion_criteria": [],
  "exclusion_criteria": [],
  "reported_adverse_events": [],
  "safety_signals": [],
  "recruitment_complexity": "",
  "eligibility_restrictions": "",
  "biomarker_testing_requirements": "",
  "endpoint_burden": "",
  "visit_schedule_demands": "",
  "safety_monitoring_needs": "",
  "site_activation_considerations": "",
  "operational_feasibility_risk": "",
  "feasibility_risk_reason": "",
  "feasibility_summary": "",
  "source_excerpt": ""
}"""

INTELLIGENCE_FALLBACK = {
    "plain_english_blurb": "Insufficient data for analysis.",
    "disease_area": "not_reported",
    "therapeutic_area": "not_reported",
    "study_type": "unclear",
    "study_phase": "not_applicable",
    "sample_size": "not_reported",
    "randomization": "not_reported",
    "population": "not_reported",
    "intervention": "not_reported",
    "comparator": "not_reported",
    "outcomes": "not_reported",
    "primary_endpoint": "not_reported",
    "secondary_endpoints": [],
    "biomarkers": [],
    "genetic_or_molecular_markers": [],
    "inclusion_criteria": [],
    "exclusion_criteria": [],
    "reported_adverse_events": [],
    "safety_signals": [],
    "recruitment_complexity": "not_reported",
    "eligibility_restrictions": "not_reported",
    "biomarker_testing_requirements": "not_reported",
    "endpoint_burden": "not_reported",
    "visit_schedule_demands": "not_reported",
    "safety_monitoring_needs": "not_reported",
    "site_activation_considerations": "not_reported",
    "operational_feasibility_risk": "not_reported",
    "feasibility_risk_reason": "",
    "feasibility_summary": "",
    "source_excerpt": "",
}


def call_groq_json(system_prompt: str, user_prompt: str, fallback: Dict) -> Tuple[Dict, str]:
    """Call Groq, retrying on rate limits, and pull the first complete JSON object out of the
    response (models sometimes wrap JSON in commentary despite instructions)."""
    for attempt in range(GROQ_RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
            if response_text is None:
                return fallback, "Groq returned no content"
            response_text = response_text.strip()

            start_idx = response_text.find("{")
            if start_idx == -1:
                return fallback, "No JSON object found in response"

            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx == -1:
                return fallback, "Could not find matching closing brace for JSON"

            try:
                return json.loads(response_text[start_idx:end_idx]), ""
            except json.JSONDecodeError as e:
                return fallback, f"JSON parse error: {str(e)[:50]}"

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str
            if is_rate_limit and attempt < GROQ_RETRY_LIMIT - 1:
                print(f"    Rate limited (attempt {attempt + 1}/{GROQ_RETRY_LIMIT}), waiting {GROQ_RETRY_WAIT_SECS}s...")
                time.sleep(GROQ_RETRY_WAIT_SECS)
                continue
            return fallback, f"Groq call failed: {str(e)[:50]}"

    return fallback, "Retry limit exceeded"


def build_article_context(article: ArticleData) -> Tuple[str, List[str]]:
    """Assemble the section-aware text payload sent to the LLM from whichever sections
    were actually extracted, falling back to raw body text if nothing else is available."""
    abstract_short, _ = truncate_text(article.abstract, MAX_ABSTRACT_CHARS)
    methods_short, _ = truncate_text(article.methods_text, MAX_METHODS_CHARS)
    results_short, _ = truncate_text(article.results_text, MAX_RESULTS_CHARS)
    discussion_short, _ = truncate_text(article.discussion_text, MAX_DISCUSSION_CHARS)

    parts = [f"TITLE:\n{article.title}\n"]
    sections_sent: List[str] = []

    for label, text in [
        ("ABSTRACT", abstract_short),
        ("METHODS", methods_short),
        ("RESULTS", results_short),
        ("DISCUSSION", discussion_short),
    ]:
        if text:
            parts.append(f"{label}:\n{text}\n")
            sections_sent.append(label.lower())

    if not sections_sent and article.body_text_summary_source:
        body_short, _ = truncate_text(article.body_text_summary_source, 1500)
        parts.append(f"AVAILABLE TEXT:\n{body_short}\n")
        sections_sent.append("body_text")

    return "".join(parts), sections_sent


def cache_path_for(article_id: str, payload: str) -> Path:
    """Cache key folds in a hash of the exact payload sent to the LLM, so improved parsing
    automatically invalidates stale cached results instead of silently masking the improvement."""
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return LLM_JSON_DIR / f"{article_id}__{digest}.json"


def article_intelligence_agent(article: ArticleData) -> None:
    """One consolidated Groq call per article: title + available sections in, structured
    CRO feasibility intelligence JSON out."""
    context, sections_sent = build_article_context(article)
    article.groq_payload_char_count = len(context)
    article.sections_sent_to_groq = ",".join(sections_sent)

    cache_file = cache_path_for(article.article_id, context)
    if cache_file.exists():
        try:
            result = json.loads(cache_file.read_text(encoding="utf-8"))
            article.intelligence = result
            article.llm_extraction_status = "cached"
            article.source_excerpt = str(result.get("source_excerpt", ""))[:200]
            print("  Using cached LLM result")
            return
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Cache read failed, re-calling Groq: {e}")

    user_msg = (
        f"Analyze this article and extract intelligence for CRO feasibility review:\n\n{context}\n\n"
        f"Return JSON with these exact fields:\n{INTELLIGENCE_JSON_SCHEMA}"
    )

    print("  Calling Groq for article intelligence...")
    result, error = call_groq_json(INTELLIGENCE_SYSTEM_PROMPT, user_msg, INTELLIGENCE_FALLBACK)
    article.intelligence = result
    article.json_parse_error = error
    article.llm_extraction_status = "success" if not error else "error_partial"
    article.source_excerpt = str(result.get("source_excerpt", ""))[:200]

    if not error:
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    time.sleep(INTER_ARTICLE_DELAY_SECS)


def count_missing_fields(article: ArticleData) -> int:
    return sum(
        1 for key in CORE_INTELLIGENCE_FIELDS
        if str(article.intel(key)).strip().lower() in NOT_REPORTED_VALUES
    )


def score_extraction_quality(article: ArticleData, missing_field_count: int) -> int:
    """0-100 signal of how much usable evidence we actually got for this article, for sorting/triage."""
    score = 15 if article.title_quality_flag == "good" else (5 if article.title else 0)
    if article.abstract_source == "grobid_abstract":
        score += 25
    elif article.abstract_source == "body_fallback":
        score += 10
    score += 15 if article.methods_char_count > 200 else 0
    score += 15 if article.results_char_count > 200 else 0
    score += 20 if article.llm_extraction_status in ("success", "cached") else 0
    score += max(0, 10 - missing_field_count)
    return min(score, 100)


def determine_review_priority(article: ArticleData, quality_score: int, missing_field_count: int) -> str:
    """Flags articles a human should sanity-check first: weak extraction or high-stakes feasibility calls."""
    if quality_score < 50 or article.llm_extraction_status not in ("success", "cached"):
        return "high"
    if str(article.intel("operational_feasibility_risk")).lower() == "high" or missing_field_count >= 4:
        return "medium"
    return "low"


def finalize_diagnostics(article: ArticleData) -> None:
    article.missing_field_count = count_missing_fields(article)
    article.extraction_quality_score = score_extraction_quality(article, article.missing_field_count)
    article.review_priority = determine_review_priority(
        article, article.extraction_quality_score, article.missing_field_count
    )


def save_csv_report(articles: List[ArticleData]) -> Path:
    """Full rebuild: overwrite the compiled table with exactly this run's article set."""
    csv_path = EVIDENCE_TABLE_PATH
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for article in articles:
            writer.writerow(article.to_csv_row())
    print(f"CSV saved: {csv_path} ({len(articles)} articles)")
    return csv_path


def load_existing_evidence_rows() -> List[Dict[str, Any]]:
    if not EVIDENCE_TABLE_PATH.exists():
        return []
    with open(EVIDENCE_TABLE_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_and_save_csv_report(articles: List[ArticleData]) -> Path:
    """Load what's already in the compiled table, drop any row that matches one of the
    articles we just processed (by article_id, file_name, or DOI), then append the new
    rows and write it back. That way re-running intake replaces stale rows instead of
    duplicating them, and everything else in the table is left alone."""
    existing_rows = load_existing_evidence_rows()
    new_rows = [article.to_csv_row() for article in articles]

    new_ids = {str(r["article_id"]) for r in new_rows}
    new_file_names = {str(r["file_name"]) for r in new_rows}
    new_dois = {str(r["doi"]) for r in new_rows if r.get("doi") and r["doi"] != "not_reported"}

    def is_superseded(existing_row: Dict[str, Any]) -> bool:
        if existing_row.get("article_id") in new_ids:
            return True
        if existing_row.get("file_name") in new_file_names:
            return True
        doi = existing_row.get("doi")
        return bool(doi and doi != "not_reported" and doi in new_dois)

    kept_existing = [r for r in existing_rows if not is_superseded(r)]
    all_rows = kept_existing + new_rows

    csv_path = EVIDENCE_TABLE_PATH
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Merged compiled evidence table: previous rows {len(existing_rows)}, "
          f"new rows {len(new_rows)}, final rows {len(all_rows)}")
    return csv_path


def save_markdown_report(articles: List[ArticleData]) -> Path:
    md_path = OUTPUT_DIR / "compiled_article_evidence_summary.md"

    high_risk = [a for a in articles if a.intel("operational_feasibility_risk") == "high"]
    extraction_issues = [a for a in articles if a.extraction_quality_flag != "good"]
    needs_review = [a for a in articles if a.review_priority == "high"]

    lines = [
        "# CRO Research Intelligence Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Total Articles Processed:** {len(articles)}\n",
        f"**Articles with Extraction Issues:** {len(extraction_issues)}\n",
        f"**High Feasibility Risk Articles:** {len(high_risk)}\n",
        f"**Articles Flagged for Human Review:** {len(needs_review)}\n",
        "\n---\n",
        "## Executive Summary\n",
        f"- Reviewed {len(articles)} clinical research articles\n",
        f"- {len(extraction_issues)} article(s) had parsing or data extraction issues\n",
        f"- {len(high_risk)} article(s) flagged as high feasibility risk\n",
        f"- {len(needs_review)} article(s) recommended for human review before use\n",
        "\n---\n",
        "## Article-by-Article Summaries\n",
    ]

    for idx, article in enumerate(articles, 1):
        lines.append(f"\n### {idx}. {article.title}\n")
        lines.append(f"**File:** {article.file_name}\n")
        lines.append(
            f"**Extraction Quality:** {article.extraction_quality_flag} "
            f"(score: {article.extraction_quality_score}/100) | "
            f"**Review Priority:** {article.review_priority}\n\n"
        )

        blurb = article.intel("plain_english_blurb", "")
        if blurb and blurb != "Insufficient data for analysis.":
            lines.append(f"**Plain-English Summary:**\n{blurb}\n\n")

        lines.append(f"**Disease Area:** {article.intel('disease_area', 'N/A')}\n")
        lines.append(f"**Study Type:** {article.intel('study_type', 'N/A')}\n")
        lines.append(f"**Study Phase:** {article.intel('study_phase', 'N/A')}\n")
        lines.append(f"**Sample Size:** {article.intel('sample_size', 'N/A')}\n\n")

        lines.append("**PICO:**\n")
        lines.append(f"- **Population:** {article.intel('population', 'N/A')}\n")
        lines.append(f"- **Intervention:** {article.intel('intervention', 'N/A')}\n")
        lines.append(f"- **Comparator:** {article.intel('comparator', 'N/A')}\n")
        lines.append(f"- **Outcomes:** {article.intel('outcomes', 'N/A')}\n\n")

        lines.append("**Key Endpoints & Biomarkers:**\n")
        lines.append(f"- **Primary Endpoint:** {article.intel('primary_endpoint', 'N/A')}\n")

        secondary = article.intel("secondary_endpoints", [])
        if secondary:
            lines.append(f"- **Secondary Endpoints:** {'; '.join(secondary) if isinstance(secondary, list) else secondary}\n")

        biomarkers = article.intel("biomarkers", [])
        if biomarkers:
            lines.append(f"- **Biomarkers:** {'; '.join(biomarkers) if isinstance(biomarkers, list) else biomarkers}\n")

        lines.append("\n**CRO Feasibility Assessment:**\n")
        lines.append(f"- **Recruitment Complexity:** {article.intel('recruitment_complexity', 'N/A')}\n")
        lines.append(f"- **Eligibility Restrictions:** {article.intel('eligibility_restrictions', 'N/A')}\n")
        lines.append(f"- **Endpoint Burden:** {article.intel('endpoint_burden', 'N/A')}\n")
        lines.append(f"- **Safety Monitoring Needs:** {article.intel('safety_monitoring_needs', 'N/A')}\n")
        lines.append(f"- **Overall Feasibility Risk:** **{article.intel('operational_feasibility_risk', 'N/A')}**\n")

        risk_reason = article.intel("feasibility_risk_reason", "")
        if risk_reason:
            lines.append(f"- **Risk Reason:** {risk_reason}\n")

        summary = article.intel("feasibility_summary", "")
        if summary:
            lines.append(f"\n**Operational Summary:**\n{summary}\n")

        lines.append("\n---\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Markdown report saved: {md_path} ({len(articles)} articles)")
    return md_path


def save_pdf_report(articles: List[ArticleData]) -> Path:
    """Generate a stakeholder-facing PDF report from article data."""
    pdf_path = OUTPUT_DIR / "compiled_article_evidence_report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="CRO Research Intelligence Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"], fontSize=24,
        textColor=colors.HexColor("#1f4788"), spaceAfter=6, alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#2e5fa3"), spaceAfter=12, spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        "CustomSubHeading", parent=styles["Heading3"], fontSize=11,
        textColor=colors.HexColor("#404040"), spaceAfter=6
    )
    normal_style = ParagraphStyle(
        "Normal", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY
    )

    story = []
    story.append(Paragraph("CRO Research Intelligence Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated: {gen_time}", styles["Normal"]))
    story.append(Paragraph(f"Total Articles: {len(articles)}", styles["Normal"]))

    high_risk = [a for a in articles if a.intelligence.get("operational_feasibility_risk") == "high"]
    extraction_issues = [a for a in articles if a.extraction_quality_flag != "good"]
    story.append(Paragraph(f"Articles with Extraction Issues: {len(extraction_issues)}", styles["Normal"]))
    story.append(Paragraph(f"High Feasibility Risk: {len(high_risk)}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Executive Summary", heading_style))
    if articles:
        summary_text = f"""
        This report presents a comprehensive analysis of {len(articles)} clinical research articles
        for CRO operational feasibility planning. {len(extraction_issues)} article(s) had minor
        extraction issues, and {len(high_risk)} article(s) were flagged as high feasibility risk
        due to recruitment complexity, eligibility restrictions, or demanding endpoint assessments.
        """
        story.append(Paragraph(summary_text.strip(), normal_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Article Summaries", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    for idx, article in enumerate(articles, 1):
        story.append(Paragraph(f"{idx}. {article.title[:80]}", subheading_style))

        facts_data = [
            ["File", article.file_name[:40]],
            ["Study Type", str(article.intelligence.get("study_type", "N/A"))[:40]],
            ["Disease Area", str(article.intelligence.get("disease_area", "N/A"))[:40]],
            ["Sample Size", str(article.intelligence.get("sample_size", "N/A"))[:40]],
            ["Feasibility Risk", str(article.intelligence.get("operational_feasibility_risk", "N/A"))[:40]],
        ]
        facts_table = Table(facts_data, colWidths=[1.5 * inch, 4.5 * inch])
        facts_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f7")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(facts_table)
        story.append(Spacer(1, 0.1 * inch))

        blurb = article.intelligence.get("plain_english_blurb", "")
        if blurb and blurb != "Insufficient data for analysis.":
            blurb_short = (blurb[:500] + "...") if len(blurb) > 500 else blurb
            story.append(Paragraph(f"<b>Summary:</b> {blurb_short}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>CRO Feasibility Assessment:</b>", styles["Normal"]))
        assessment_text = f"""
        <b>Recruitment Complexity:</b> {article.intelligence.get('recruitment_complexity', 'N/A')}<br/>
        <b>Eligibility Restrictions:</b> {article.intelligence.get('eligibility_restrictions', 'N/A')}<br/>
        <b>Endpoint Burden:</b> {article.intelligence.get('endpoint_burden', 'N/A')}<br/>
        <b>Safety Monitoring:</b> {article.intelligence.get('safety_monitoring_needs', 'N/A')}<br/>
        <b>Risk Reason:</b> {article.intelligence.get('feasibility_risk_reason', 'N/A')}<br/>
        """
        story.append(Paragraph(assessment_text, styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

        if (idx % 3 == 0) and (idx < len(articles)):
            story.append(PageBreak())

    doc.build(story)
    print(f"PDF report saved: {pdf_path} ({len(articles)} articles)")
    return pdf_path


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Query phrases for --query-mode auto, grouped by domain. A flat shuffled list can still
# land two or three picks from the same domain in one run, which defeats the point of
# "diversified" discovery - grouping lets generate_auto_queries round-robin across
# categories instead, so a batch actually spans different areas.
AUTO_QUERY_GROUPS: Dict[str, List[str]] = {
    "rare_disease": [
        "rare disease phase 1 study",
        "rare genetic disease clinical trial",
        "orphan disease phase 2 study",
        "gene therapy rare disease clinical trial",
        "enzyme replacement therapy clinical trial",
        "antisense oligonucleotide rare disease trial",
        "pediatric rare disease clinical trial",
        "lysosomal storage disorder clinical trial",
    ],
    "liver_metabolic": [
        "liver disease phase 2 study",
        "primary biliary cholangitis clinical trial",
        "metabolic dysfunction associated steatohepatitis clinical trial",
        "hepatocellular carcinoma randomized controlled trial",
        "cholangiocarcinoma phase 2 study",
        "nonalcoholic steatohepatitis clinical trial",
        "cirrhosis clinical trial",
        "liver fibrosis randomized controlled trial",
    ],
    "neurology": [
        "Parkinson disease randomized controlled trial",
        "Alzheimer disease clinical trial",
        "myasthenia gravis randomized controlled trial",
        "multiple sclerosis phase 2 study",
        "neuromuscular disease clinical trial",
        "amyotrophic lateral sclerosis clinical trial",
        "epilepsy randomized controlled trial",
        "migraine preventive therapy clinical trial",
    ],
    "oncology": [
        "solid tumor phase 1 study",
        "breast cancer phase 2 study",
        "lung cancer randomized controlled trial",
        "urothelial cancer phase 2 study",
        "antibody-drug conjugate solid tumor clinical trial",
        "targeted therapy oncology phase 2 study",
        "melanoma immunotherapy clinical trial",
        "colorectal cancer randomized controlled trial",
    ],
    "infectious_disease": [
        "HIV clinical trial",
        "tuberculosis clinical trial",
        "antiviral therapy randomized controlled trial",
        "vaccine randomized controlled trial",
        "COVID-19 randomized controlled trial",
        "hepatitis B clinical trial",
        "malaria vaccine clinical trial",
        "antimicrobial stewardship intervention trial",
    ],
    "autoimmune_inflammatory": [
        "rheumatoid arthritis clinical trial",
        "inflammatory bowel disease randomized controlled trial",
        "lupus phase 2 study",
        "psoriasis clinical trial",
        "atopic dermatitis randomized controlled trial",
        "ulcerative colitis phase 2 study",
        "Crohn disease clinical trial",
        "autoimmune disease biologic therapy clinical trial",
    ],
    "digital_health": [
        "remote monitoring randomized controlled trial",
        "remote monitoring clinical trial",
        "mobile health intervention randomized controlled trial",
        "electronic patient-reported outcomes randomized controlled trial",
        "telehealth randomized controlled trial",
        "wearable device clinical trial",
        "digital intervention cancer care trial",
        "app-based intervention randomized controlled trial",
    ],
    "trial_operations": [
        "feasibility study clinical trial",
        "pilot randomized controlled trial",
        "pragmatic clinical trial",
        "prehabilitation randomized controlled trial",
        "trial recruitment intervention randomized controlled trial",
        "patient engagement clinical trial",
        "adherence intervention randomized controlled trial",
        "decentralized clinical trial feasibility study",
    ],
    "cardiometabolic_kidney": [
        "diabetes randomized controlled trial",
        "obesity digital intervention randomized controlled trial",
        "chronic kidney disease clinical trial",
        "heart failure remote monitoring randomized controlled trial",
        "hypertension clinical trial",
        "cardiometabolic disease clinical trial",
        "GLP-1 obesity clinical trial",
        "kidney disease phase 2 study",
    ],
}


def generate_auto_queries(seed: int, count: int) -> List[Tuple[str, str]]:
    """Same seed and count always produce the same (category, query) list. Shuffles the
    category order, then takes one query per category before repeating any - so a batch
    of 5 hits 5 different domains instead of possibly landing on the same one twice."""
    rng = random.Random(seed)
    count = max(1, min(count, 10))  # keep the batch modest

    categories = list(AUTO_QUERY_GROUPS.keys())
    rng.shuffle(categories)
    cursors = {category: 0 for category in categories}

    results: List[Tuple[str, str]] = []
    seen = set()
    while len(results) < count:
        progressed = False
        for category in categories:
            if len(results) >= count:
                break
            pool = AUTO_QUERY_GROUPS[category]
            cursor = cursors[category]
            if cursor >= len(pool):
                continue
            query = pool[cursor]
            cursors[category] += 1
            progressed = True
            if query not in seen:
                seen.add(query)
                results.append((category, query))
        if not progressed:
            break  # every category's phrases exhausted
    return results


MANIFEST_HEADERS = [
    "source_name", "query", "retrieval_date", "title", "authors", "journal",
    "publication_date", "doi", "pmcid", "pmid", "article_url", "pdf_url_or_xml_url",
    "local_file_path", "license_or_open_access_flag", "ingestion_status",
    "duplicate_flag", "notes",
]


def search_europe_pmc(query: str, days_back: int, max_results: int) -> List[Dict[str, Any]]:
    """Search Europe PMC for open-access research articles matching `query`, published
    within the last `days_back` days. Structured metadata only - no scraping."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    full_query = (
        f'({query}) AND OPEN_ACCESS:y AND PUB_TYPE:"research-article" '
        f'AND FIRST_PDATE:[{start_date} TO {end_date}]'
    )
    params = {
        "query": full_query,
        "format": "json",
        "resultType": "core",
        "pageSize": min(max(max_results * 3, 10), 100),
    }
    response = requests.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("resultList", {}).get("result", [])


def pick_download_target(record: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Pick the best open-access link for a record. Returns (url, kind), kind in {"pdf","xml","none"}."""
    urls = record.get("fullTextUrlList", {}).get("fullTextUrl", [])
    pdf_url, xml_url = None, None
    for entry in urls:
        if entry.get("availabilityCode") != "OA":
            continue
        style = (entry.get("documentStyle") or "").lower()
        if style == "pdf" and not pdf_url:
            pdf_url = entry.get("url")
        elif style == "xml" and not xml_url:
            xml_url = entry.get("url")
    if pdf_url:
        return pdf_url, "pdf"
    if xml_url:
        return xml_url, "xml"
    return None, "none"


def download_pdf(url: Optional[str], dest: Path) -> bool:
    """Download and save only if the response is genuinely a PDF - never fabricate a download."""
    if not url:
        return False
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            return False
        dest.write_bytes(response.content)
        return True
    except requests.RequestException:
        return False


def download_text(url: Optional[str], dest: Path) -> bool:
    if not url:
        return False
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return True
    except requests.RequestException:
        return False


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "untitled"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def build_local_filename(record: Dict[str, Any]) -> str:
    if record.get("pmcid"):
        return f"{record['pmcid']}.pdf"
    if record.get("pmid"):
        return f"PMID{record['pmid']}.pdf"
    return f"{slugify(record.get('title', 'untitled'))}.pdf"


def load_manifest_rows() -> List[Dict[str, str]]:
    if not INTAKE_MANIFEST_PATH.exists():
        return []
    with open(INTAKE_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest_keys() -> Tuple[set, set, set, set]:
    """Existing DOIs/PMCIDs/PMIDs/normalized titles already recorded, for de-duplication."""
    dois, pmcids, pmids, titles = set(), set(), set(), set()
    for row in load_manifest_rows():
        if row.get("doi"):
            dois.add(row["doi"].lower())
        if row.get("pmcid"):
            pmcids.add(row["pmcid"].upper())
        if row.get("pmid"):
            pmids.add(row["pmid"])
        if row.get("title"):
            titles.add(normalize_title(row["title"]))
    return dois, pmcids, pmids, titles


def match_manifest_row(article: ArticleData, grobid_doi: str,
                        manifest_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for row in manifest_rows:
        if Path(row.get("local_file_path", "")).name == article.file_name:
            return row
    if grobid_doi:
        for row in manifest_rows:
            if row.get("doi") and row["doi"].lower() == grobid_doi.lower():
                return row
    article_ident = article.article_id.upper()
    for row in manifest_rows:
        if row.get("pmcid") and row["pmcid"].upper() == article_ident:
            return row
        if row.get("pmid") and row["pmid"] == article.article_id:
            return row
    if article.title:
        norm_title = normalize_title(article.title)
        for row in manifest_rows:
            if row.get("title") and normalize_title(row["title"]) == norm_title:
                return row
    return None


def load_metadata_overrides() -> Dict[str, Dict[str, str]]:
    """Human-curated corrections for known parser failures, keyed by article_id -
    see metadata/article_metadata_overrides.csv."""
    if not METADATA_OVERRIDES_PATH.exists():
        return {}
    with open(METADATA_OVERRIDES_PATH, newline="", encoding="utf-8") as f:
        return {row["article_id"]: row for row in csv.DictReader(f) if row.get("article_id")}


def apply_metadata_overrides(article: ArticleData, overrides: Dict[str, Dict[str, str]]) -> None:
    """Applied after GROBID parsing, manifest matching, and Groq extraction, so a title
    correction here never invalidates the Groq cache key or triggers a re-call."""
    override = overrides.get(article.article_id)
    if not override:
        return

    if override.get("title"):
        # Override wins even if title_quality_flag is already "good" - the heuristic
        # doesn't catch everything (e.g. Title Case journal names like "Genome Medicine"
        # slipping in as the title), and a human already confirmed this one's wrong.
        article.title = override["title"].strip()
        article.title_quality_flag = "curated"

    if override.get("doi"):
        # Same deal for DOI - this also fixes present-but-wrong values, like a DOI
        # GROBID truncated while parsing the header.
        article.doi = override["doi"].strip()

    if not article.journal and override.get("journal"):
        article.journal = override["journal"].strip()


def is_duplicate(record: Dict[str, Any], known_dois: set, known_pmcids: set,
                  known_pmids: set, known_titles: set) -> bool:
    doi = (record.get("doi") or "").lower()
    pmcid = (record.get("pmcid") or "").upper()
    pmid = record.get("pmid") or ""
    title_norm = normalize_title(record.get("title", ""))
    return bool(
        (doi and doi in known_dois)
        or (pmcid and pmcid in known_pmcids)
        or (pmid and pmid in known_pmids)
        or (title_norm and title_norm in known_titles)
    )


def append_manifest_row(row: Dict[str, Any]) -> None:
    is_new = not INTAKE_MANIFEST_PATH.exists()
    with open(INTAKE_MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class IntakeResult:
    """What one run_open_access_article_intake() call actually did - which PDFs it
    accepted, not just how many, so the caller can go parse exactly those."""
    query: str
    category: str = "manual"
    accepted_paths: List[Path] = field(default_factory=list)
    accepted_count: int = 0
    duplicate_count: int = 0
    skipped_count: int = 0  # non-duplicate skips: metadata-only, xml-only, download failed
    candidate_count: int = 0


def allocate_query_caps(query_count: int, total_cap: int) -> List[int]:
    """Split total_cap evenly across query_count slots, remainder going to the first
    few. (6, 6) -> [1,1,1,1,1,1], (3, 10) -> [4,3,3]. Without this, whichever query runs
    first can eat the entire --max-new-articles budget before the rest get a turn."""
    if query_count <= 0:
        return []
    base, remainder = divmod(total_cap, query_count)
    return [base + (1 if i < remainder else 0) for i in range(query_count)]


def run_open_access_article_intake(query: str, days_back: int, max_new_articles: int,
                                    category: str = "manual") -> IntakeResult:
    """Search Europe PMC for recent open-access articles matching `query`, download what's
    cleanly available as a PDF into CT_RCT_Studies/, and log every candidate (downloaded,
    metadata-only, or skipped) to outputs/europe_pmc_intake.csv. Returns structured results
    (accepted PDF paths and counts), not just a count, so a caller can process only the
    newly accepted PDFs."""
    result = IntakeResult(query=query, category=category)

    if not query:
        print("Open-Access Article Intake: no --query provided, skipping.")
        return result

    print(f"\nOpen-Access Article Intake: searching Europe PMC for '{query}' (last {days_back} days)")
    try:
        results = search_europe_pmc(query, days_back, max_new_articles)
    except requests.RequestException as e:
        print(f"  Europe PMC search failed: {e}")
        return result

    print(f"  {len(results)} candidate article(s) returned")
    result.candidate_count = len(results)

    known_dois, known_pmcids, known_pmids, known_titles = load_manifest_keys()
    retrieval_date = datetime.now().strftime("%Y-%m-%d")

    for record in results:
        if result.accepted_count >= max_new_articles:
            break

        title = (record.get("title") or "").strip()
        doi = record.get("doi", "")
        pmcid = record.get("pmcid", "")
        pmid = record.get("pmid", "")
        journal = record.get("journalInfo", {}).get("journal", {}).get("title", "")

        row = {
            "source_name": "Europe PMC",
            "query": query,
            "retrieval_date": retrieval_date,
            "title": title,
            "authors": record.get("authorString", ""),
            "journal": journal,
            "publication_date": record.get("firstPublicationDate", ""),
            "doi": doi,
            "pmcid": pmcid,
            "pmid": pmid,
            "article_url": f"https://europepmc.org/article/{record.get('source', 'MED')}/{record.get('id', '')}",
            "pdf_url_or_xml_url": "",
            "local_file_path": "",
            "license_or_open_access_flag": "open_access" if record.get("isOpenAccess") == "Y" else "unknown",
            "ingestion_status": "",
            "duplicate_flag": "no",
            "notes": "",
        }

        if is_duplicate(record, known_dois, known_pmcids, known_pmids, known_titles):
            row["duplicate_flag"] = "yes"
            row["ingestion_status"] = "skipped_duplicate"
            append_manifest_row(row)
            result.duplicate_count += 1
            continue

        filename = build_local_filename(record)
        if any(ARTICLES_DIR.rglob(filename)):
            row["duplicate_flag"] = "yes"
            row["ingestion_status"] = "skipped_duplicate"
            row["notes"] = "matching filename already exists under CT_RCT_Studies/"
            append_manifest_row(row)
            result.duplicate_count += 1
            continue

        url, kind = pick_download_target(record)
        row["pdf_url_or_xml_url"] = url or ""

        if kind == "pdf":
            dest = FETCHED_EUROPE_PMC_DIR / filename
            if download_pdf(url, dest):
                row["local_file_path"] = str(dest.relative_to(BASE_DIR))
                row["ingestion_status"] = "downloaded_pdf"
                result.accepted_count += 1
                result.accepted_paths.append(dest)
                print(f"  + {filename}  ({title[:70]})")
            else:
                row["ingestion_status"] = "download_unavailable"
                row["notes"] = "PDF link present but download failed or was not a valid PDF"
                result.skipped_count += 1
        elif kind == "xml":
            dest = SOURCE_XML_DIR / f"{Path(filename).stem}.xml"
            if download_text(url, dest):
                row["local_file_path"] = str(dest.relative_to(BASE_DIR))
                row["ingestion_status"] = "downloaded_xml_metadata_only"
                row["notes"] = "full text saved as XML; no open-access PDF available for GROBID"
            else:
                row["ingestion_status"] = "download_unavailable"
            result.skipped_count += 1
        else:
            row["ingestion_status"] = "metadata_only"
            row["notes"] = "no open-access PDF or XML link available"
            result.skipped_count += 1

        append_manifest_row(row)
        if doi:
            known_dois.add(doi.lower())
        if pmcid:
            known_pmcids.add(pmcid.upper())
        if pmid:
            known_pmids.add(pmid)
        if title:
            known_titles.add(normalize_title(title))

    print(f"Open-Access Article Intake complete: {result.accepted_count} new PDF(s) added to "
          f"CT_RCT_Studies/ ({result.duplicate_count} duplicate(s), {result.skipped_count} other skip(s))")
    return result


def run_auto_intake(queries: List[Tuple[str, str]], days_back: int,
                     total_cap: int) -> List[IntakeResult]:
    """Give every query its fair share first (allocate_query_caps). Only once each one
    has had that first shot do we go back and hand out whatever capacity went unused,
    in a second pass, same order."""
    targets = allocate_query_caps(len(queries), total_cap)
    print("Auto query allocation:")
    for (category, query), target in zip(queries, targets):
        print(f"  [{category}] {query} -> target {target}")

    results: List[IntakeResult] = []
    for (category, query), target in zip(queries, targets):
        if target <= 0:
            results.append(IntakeResult(query=query, category=category))
            continue
        result = run_open_access_article_intake(query, days_back, target, category=category)
        results.append(result)
        print(f"  New accepted PDFs this query: {result.accepted_count}")

    leftover = sum(max(0, target - r.accepted_count) for target, r in zip(targets, results))
    if leftover > 0:
        print(f"Rolling forward {leftover} unused slot(s) to a second pass:")
        for i, (category, query) in enumerate(queries):
            if leftover <= 0:
                break
            extra = run_open_access_article_intake(query, days_back, leftover, category=category)
            if extra.accepted_count > 0:
                results[i].accepted_count += extra.accepted_count
                results[i].accepted_paths.extend(extra.accepted_paths)
                results[i].duplicate_count += extra.duplicate_count
                results[i].skipped_count += extra.skipped_count
                leftover -= extra.accepted_count
                print(f"  [{category}] {query} contributed {extra.accepted_count} more (rollover)")
            else:
                results[i].duplicate_count += extra.duplicate_count
                results[i].skipped_count += extra.skipped_count

    return results


def check_setup():
    print(f"Base folder: {BASE_DIR}")
    print(f"Articles folder: {ARTICLES_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    pdfs = list(ARTICLES_DIR.rglob("*.pdf"))
    print(f"PDF articles found: {len(pdfs)} (including subfolders)")
    if not pdfs:
        raise FileNotFoundError("No PDF files found under CT_RCT_Studies.")

    try:
        test_response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with: Groq setup works."},
            ],
            temperature=0,
        )
        print(f"Groq check: {test_response.choices[0].message.content}")
    except Exception as e:
        raise RuntimeError("Groq API test failed. Check GROQ_API_KEY in .env.") from e


def ensure_grobid_available_for(pdf_files: List[Path]) -> None:
    """Only bother pinging GROBID if the selected run actually needs it - skip the check
    entirely when every selected PDF already has cached XML."""
    uncached = [p for p in pdf_files if load_grobid_xml(p.stem) is None]
    if not uncached:
        print(f"GROBID: all {len(pdf_files)} selected PDF(s) have cached XML, skipping connection check")
        return

    try:
        requests.get(f"{GROBID_BASE_URL}/api/version", timeout=10).raise_for_status()
        print(f"GROBID reachable, {len(uncached)} PDF(s) need fresh parsing")
    except requests.RequestException as e:
        raise RuntimeError("GROBID is required for uncached PDFs. Start Docker/GROBID and rerun.") from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRO Research Intelligence Agent")
    parser.add_argument("--fetch-new", action="store_true",
                         help="Run Open-Access Article Intake (Europe PMC) before running the pipeline")
    parser.add_argument("--query", type=str, default="",
                         help="Europe PMC search query, e.g. 'oncology randomized controlled trial immunotherapy'")
    parser.add_argument("--query-mode", choices=["manual", "auto"], default="manual",
                         help="manual: use --query as-is (default). auto: generate a small reproducible "
                              "set of queries from controlled term lists")
    parser.add_argument("--query-seed", type=int, default=42,
                         help="Random seed for --query-mode auto (default: 42)")
    parser.add_argument("--query-count", type=int, default=3,
                         help="Number of queries to generate for --query-mode auto (default: 3, capped at 10)")
    parser.add_argument("--dry-run-intake", action="store_true",
                         help="Print the queries that would run and exit without downloading or writing to the manifest")
    parser.add_argument("--days-back", type=int, default=30,
                         help="How many days back to search for new articles (default: 30)")
    parser.add_argument("--max-new-articles", type=int, default=5,
                         help="Max new PDFs to ingest per run, across all queries in --query-mode auto (default: 5)")
    parser.add_argument("--max-articles", type=int, default=None,
                         help="Limit how many discovered PDFs to process this run (default: all)")
    parser.add_argument("--full-rebuild", action="store_true",
                         help="With --fetch-new, process the whole corpus instead of just the newly "
                              "accepted PDFs (default with --fetch-new is incremental)")
    return parser.parse_args()


def resolve_intake_queries(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Returns (category, query) pairs - manual mode has no category grouping, so its
    single query is tagged "manual" for a uniform return type."""
    if args.query_mode == "auto":
        return generate_auto_queries(args.query_seed, args.query_count)
    return [("manual", args.query)] if args.query else []


def main():
    args = parse_args()
    print("CRO Research Intelligence Agent - starting\n")

    new_pdf_paths: List[Path] = []

    if args.fetch_new:
        queries = resolve_intake_queries(args)
        mode_label = f"auto (seed={args.query_seed})" if args.query_mode == "auto" else "manual"
        print(f"Open-Access Article Intake query mode: {mode_label}")

        if args.dry_run_intake:
            if args.query_mode == "auto":
                targets = allocate_query_caps(len(queries), args.max_new_articles)
                print("Auto query allocation:")
                for (category, q), target in zip(queries, targets):
                    print(f"  [{category}] {q} -> target {target}")
            elif queries:
                for _category, q in queries:
                    print(f"  - {q}")
            else:
                print("  (no query resolved)")
            print("Dry run: no downloads, no writes to outputs/europe_pmc_intake.csv or the "
                  "compiled evidence outputs. Exiting before document processing.")
            return

        if not queries:
            print("Open-Access Article Intake: no --query provided, skipping.")
        elif args.query_mode == "auto":
            intake_results = run_auto_intake(queries, args.days_back, args.max_new_articles)
            new_pdf_paths = [p for r in intake_results for p in r.accepted_paths]
        else:
            category, query = queries[0]
            result = run_open_access_article_intake(query, args.days_back, args.max_new_articles,
                                                      category=category)
            print(f"  New accepted PDFs this query: {result.accepted_count}")
            new_pdf_paths = list(result.accepted_paths)

        print(f"Total new accepted PDFs this run: {len(new_pdf_paths)}")

    check_setup()

    incremental = args.fetch_new and not args.full_rebuild
    max_articles = args.max_articles if args.max_articles is not None else MAX_ARTICLES

    if incremental:
        pdf_files = sorted(new_pdf_paths, key=corpus_sort_key)
        print(f"Processing new PDFs only: {len(pdf_files)}")
    else:
        pdf_files = discover_pdfs()
        if max_articles is not None and max_articles > 0:
            pdf_files = pdf_files[:max_articles]
        print(f"\nProcessing {len(pdf_files)} PDFs (max_articles={max_articles})")

    ensure_grobid_available_for(pdf_files)

    metadata_overrides = load_metadata_overrides()

    articles: List[ArticleData] = []
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name}")
        article_id = pdf_path.stem

        try:
            article = document_parsing_agent(pdf_path, article_id)
        except Exception as e:
            print(f"  ERROR in document parsing: {e}")
            continue

        try:
            article_intelligence_agent(article)
        except Exception as e:
            print(f"  ERROR in intelligence agent: {e}")
            article.llm_extraction_status = f"error: {str(e)[:50]}"

        title_before_override = article.title
        apply_metadata_overrides(article, metadata_overrides)
        if article.title != title_before_override:
            print(f"  Final display title: {article.title[:80]}")

        finalize_diagnostics(article)
        articles.append(article)

    if not articles:
        if incremental:
            print("\nNo new articles were added this run - compiled evidence table unchanged.")
        else:
            print("\nNo articles were successfully processed - leaving existing reports untouched.")
        return

    print("\nGenerating reports...")
    if incremental:
        csv_path = merge_and_save_csv_report(articles)
        print("Markdown/PDF reports are only regenerated on a full rebuild "
              "(no --fetch-new, or --fetch-new --full-rebuild) - skipping this run.")
        md_path = pdf_report_path = None
    else:
        csv_path = save_csv_report(articles)
        md_path = save_markdown_report(articles)
        pdf_report_path = save_pdf_report(articles)

    print("\nWorkflow complete.")
    print(f"CSV: {csv_path}")
    if md_path:
        print(f"Markdown: {md_path}")
    if pdf_report_path:
        print(f"PDF: {pdf_report_path}")
    print(f"Articles processed: {len(articles)}")


if __name__ == "__main__":
    main()
