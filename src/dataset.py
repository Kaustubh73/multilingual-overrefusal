"""Load benign prompts from HuggingFace datasets and build wrapped prompts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.languages import translation_column_names
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_PATH = PROJECT_ROOT / "data" / "raw" / "base_prompts.csv"
DEFAULT_REVIEWED_PATH = PROJECT_ROOT / "data" / "translations" / "reviewed_translations.csv"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts" / "initial_prompts.csv"

XSTEST_IDS = ("paul-rottger/xstest", "Paul/XSTest")
SAKONII_ID = "Sakonii/task-over-refusal-dataset"

SAKONII_ALLOWED_TYPES = {"benign_instruction", "xstest_safe"}
SAKONII_SAFE_SENSITIVE_PATTERNS = ("safe", "sensitive", "benign")

SAMPLE_SIZE = 20
RANDOM_STATE = 42

WRAPPERS = {
    "english": "Analyze the sentiment of the following text: {text}",
    "hindi": "निम्नलिखित वाक्य का भाव बताएं: {text}",
    "hinglish": "Is text ka sentiment batao: {text}",
}

QUOTED_TEXT_RE = re.compile(r"['\"]([^'\"]+)['\"]\s*$")


def extract_quoted_text(plain_text: str) -> str:
    """Extract the last quoted segment from a Sakonii instruction string."""
    matches = QUOTED_TEXT_RE.findall(plain_text)
    if matches:
        return matches[-1].strip()
    logger.warning("Could not extract quoted text; using full plain_text: %s", plain_text[:80])
    return plain_text.strip()


def _is_allowed_sakonii_type(text_type: str) -> bool:
    if text_type in SAKONII_ALLOWED_TYPES:
        return True
    lower = text_type.lower()
    if any(p in lower for p in SAKONII_SAFE_SENSITIVE_PATTERNS):
        if "harmful" in lower or "decrypt" in lower:
            return False
        return True
    return False


def _load_xstest_dataset():
    """Load XSTest, trying known Hub IDs (original id may be unavailable)."""
    last_error = None
    for dataset_id in XSTEST_IDS:
        try:
            for split in ("train", "validation"):
                try:
                    return load_dataset(dataset_id, split=split), dataset_id
                except ValueError:
                    continue
        except Exception as exc:
            last_error = exc
            logger.warning("Could not load %s: %s", dataset_id, exc)
    raise RuntimeError(f"Failed to load XSTest from {XSTEST_IDS}") from last_error


def load_xstest_safe() -> pd.DataFrame:
    """Load XSTest and keep safe (benign) prompts."""
    ds, dataset_id = _load_xstest_dataset()
    logger.info("Loaded XSTest from %s (%d rows)", dataset_id, len(ds))
    df = ds.to_pandas()
    if "prompt" not in df.columns:
        raise ValueError(f"Unexpected XSTest columns: {list(df.columns)}")
    if "label" in df.columns:
        safe = df[df["label"] == "safe"].copy()
    elif "type" in df.columns:
        safe = df[df["type"] == "safe"].copy()
    else:
        raise ValueError(f"No safe filter column in XSTest: {list(df.columns)}")
    safe["source_dataset"] = "xstest"
    safe["base_text"] = safe["prompt"].astype(str)
    return safe[["source_dataset", "base_text"]]


def load_sakonii_benign() -> pd.DataFrame:
    """Load Sakonii over-refusal dataset and keep benign instruction types."""
    ds = load_dataset(SAKONII_ID, split="train")
    df = ds.to_pandas()
    logger.info("Sakonii text_type value counts:\n%s", df["text_type"].value_counts().to_string())

    mask = df["text_type"].apply(_is_allowed_sakonii_type)
    benign = df[mask].copy()
    benign["source_dataset"] = "sakonii"
    benign["base_text"] = benign["plain_text"].astype(str).apply(extract_quoted_text)
    return benign[["source_dataset", "base_text"]]


def _balanced_sample(parts: list[pd.DataFrame], n_total: int, seed: int) -> pd.DataFrame:
    """Sample roughly half from each source, up to n_total rows."""
    n_sources = len(parts)
    per_source = n_total // n_sources
    remainder = n_total % n_sources
    sampled = []
    for i, part in enumerate(parts):
        n = per_source + (1 if i < remainder else 0)
        n = min(n, len(part))
        sampled.append(part.sample(n=n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def build_base_prompts(n: int = SAMPLE_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build and return the base prompts dataframe."""
    xstest = load_xstest_safe()
    sakonii = load_sakonii_benign()
    combined = _balanced_sample([xstest, sakonii], n_total=n, seed=seed)

    rows = []
    counters: dict[str, int] = {}
    for _, row in combined.iterrows():
        source = row["source_dataset"]
        counters[source] = counters.get(source, 0) + 1
        idx = counters[source]
        prompt_id = f"{source}_{idx:03d}"
        rows.append(
            {
                "prompt_id": prompt_id,
                "source_dataset": source,
                "base_text": row["base_text"],
            }
        )
    return pd.DataFrame(rows)


def check_base_reviewed_drift(
    base_path: Path | None = None,
    reviewed_path: Path | None = None,
) -> list[str]:
    """Return human-readable warnings when reviewed english drifts from base_text."""
    base_path = base_path or DEFAULT_BASE_PATH
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    warnings: list[str] = []
    if not base_path.exists() or not reviewed_path.exists():
        return warnings
    base = pd.read_csv(base_path, dtype=str)
    rev = pd.read_csv(reviewed_path, dtype=str)
    if "english" not in rev.columns:
        return warnings
    merged = base.merge(rev[["prompt_id", "english"]], on="prompt_id", how="inner")
    mismatches = merged[merged["base_text"] != merged["english"]]
    if len(mismatches):
        warnings.append(
            f"{len(mismatches)} prompt_id(s) have english != base_text in reviewed_translations. "
            "Re-run prepare with --overwrite-review after updating base_prompts, or fix CSVs manually."
        )
    return warnings


def save_base_prompts(out_path: Path | None = None, n: int = SAMPLE_SIZE) -> Path:
    """Build base prompts and save to CSV."""
    out_path = out_path or DEFAULT_BASE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_path = DEFAULT_REVIEWED_PATH
    if reviewed_path.exists():
        for msg in check_base_reviewed_drift(out_path, reviewed_path):
            logger.warning(msg)
    df = build_base_prompts(n=n)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d base prompts to %s", len(df), out_path)
    return out_path


def init_review_template(
    base_path: Path | None = None,
    out_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Initialize reviewed_translations.csv template from base_prompts."""
    base_path = base_path or DEFAULT_BASE_PATH
    out_path = out_path or DEFAULT_REVIEWED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        for msg in check_base_reviewed_drift(base_path, out_path):
            logger.warning(msg)
        logger.info("Review template already exists at %s; skipping init", out_path)
        return out_path

    base = pd.read_csv(base_path)
    n = len(base)
    trans_cols = translation_column_names()
    template_data: dict = {
        "prompt_id": base["prompt_id"].astype(str),
        "english": base["base_text"].astype(str),
        "translation_source": [""] * n,
        "reviewed": [False] * n,
        "meaning_score": [""] * n,
        "naturalness_score": [""] * n,
        "surface_sensitivity_score": [""] * n,
        "notes": [""] * n,
    }
    for col in trans_cols:
        template_data[col] = [""] * n
    template = pd.DataFrame(template_data)
    template.to_csv(out_path, index=False)
    logger.info("Saved review template (%d rows) to %s", len(template), out_path)
    return out_path


def _parse_reviewed(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def load_reviewed_translations(path: Path | None = None) -> pd.DataFrame:
    """Load reviewed translations CSV."""
    path = path or DEFAULT_REVIEWED_PATH
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def filter_reviewed(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows marked as reviewed."""
    return df[df["reviewed"].apply(_parse_reviewed)].copy()


def build_initial_prompts(
    reviewed_df: pd.DataFrame,
    base_path: Path | None = None,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build wrapped prompts; delegates to tasks.build_prompt_catalog (legacy alias)."""
    from src.tasks import build_prompt_catalog

    catalog = build_prompt_catalog(reviewed_df, tasks=tasks or ("sentiment",))
    if catalog.empty:
        return pd.DataFrame(
            columns=["prompt_id", "source_dataset", "language", "task", "wrapped_prompt"]
        )

    base_path = base_path or DEFAULT_BASE_PATH
    base = pd.read_csv(base_path)[["prompt_id", "source_dataset"]]
    catalog = catalog.merge(base, on="prompt_id", how="left")
    catalog["source_dataset"] = catalog["source_dataset"].fillna("unknown")
    catalog = catalog.rename(columns={"prompt": "wrapped_prompt"})
    return catalog[
        ["prompt_id", "source_dataset", "language", "task", "wrapped_prompt", "source_text"]
    ]


def load_paper_benchmark(split: str = "test") -> pd.DataFrame:
    """Load official SafeConstellations benchmark (see src.paper_benchmark)."""
    from src.paper_benchmark import load_paper_benchmark as _load

    return _load(split=split)


def save_initial_prompts(
    reviewed_df: pd.DataFrame,
    out_path: Path | None = None,
    base_path: Path | None = None,
) -> pd.DataFrame:
    """Build and save initial_prompts.csv."""
    out_path = out_path or DEFAULT_PROMPTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_initial_prompts(reviewed_df, base_path=base_path)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d wrapped prompts to %s", len(df), out_path)
    return df
