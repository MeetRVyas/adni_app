"""
Unified Metadata Schema — BDA Layer
=====================================
Every disease dataset is normalised into this schema before entering
the pipeline. This is the contract between the BDA layer and the ML pipeline.

Core columns (ALL diseases must populate these):
    image_path      str     Absolute path to the PNG file on disk
    label           int     Primary integer class index (0-based)
    label_name      str     Human-readable class name
    split           str     'train' | 'val' | 'test'
    disease         str     Disease tag, e.g. 'adni', 'chestxray14', 'isic2024'
    fold            int     K-fold index (0-based); -1 for datasets with fixed splits
    patient_id      str     Patient/subject identifier (synthetic if unavailable)
    source          str     Dataset origin, e.g. 'kaggle', 'huggingface', 'local'
    image_width     int     Pixel width of the stored PNG
    image_height    int     Pixel height of the stored PNG

Extended columns (optional, per-disease):
    is_multilabel   bool    True when a sample can belong to multiple classes
    additional_labels str   JSON-encoded list of extra label strings (multi-label)
    age             float   Subject age in years (if available)
    sex             str     'M' | 'F' | 'U' (unknown)
    site            str     Acquisition site / scanner id
    original_format str     'DICOM' | 'NIfTI' | 'NPZ' | 'PNG' | 'JPEG'
    checksum        str     SHA-256 hex digest of the image file
"""

from __future__ import annotations
from typing import List
import pyarrow as pa

# ── Core schema ──────────────────────────────────────────────────────────────

CORE_FIELDS: List[pa.Field] = [
    pa.field("image_path",    pa.string(),  nullable=False),
    pa.field("label",         pa.int32(),   nullable=False),
    pa.field("label_name",    pa.string(),  nullable=False),
    pa.field("split",         pa.string(),  nullable=False),
    pa.field("disease",       pa.string(),  nullable=False),
    pa.field("fold",          pa.int32(),   nullable=False),
    pa.field("patient_id",    pa.string(),  nullable=False),
    pa.field("source",        pa.string(),  nullable=False),
    pa.field("image_width",   pa.int32(),   nullable=True),
    pa.field("image_height",  pa.int32(),   nullable=True),
]

# ── Extended schema ───────────────────────────────────────────────────────────

EXTENDED_FIELDS: List[pa.Field] = [
    pa.field("is_multilabel",     pa.bool_(),   nullable=True),
    pa.field("additional_labels", pa.string(),  nullable=True),   # JSON list
    pa.field("age",               pa.float32(), nullable=True),
    pa.field("sex",               pa.string(),  nullable=True),
    pa.field("site",              pa.string(),  nullable=True),
    pa.field("original_format",   pa.string(),  nullable=True),
    pa.field("checksum",          pa.string(),  nullable=True),
]

FULL_SCHEMA = pa.schema(CORE_FIELDS + EXTENDED_FIELDS)

# ── Valid splits / disease tags ──────────────────────────────────────────────

VALID_SPLITS    = {"train", "val", "test"}
VALID_SOURCES   = {"kaggle", "huggingface", "local", "nihcc", "isic", "medmnist"}

# ── Defaults for optional fields ─────────────────────────────────────────────

EXTENDED_DEFAULTS = {
    "is_multilabel":     False,
    "additional_labels": "[]",
    "age":               None,
    "sex":               "U",
    "site":              "unknown",
    "original_format":   "PNG",
    "checksum":          None,
}


def validate_row(row: dict) -> list[str]:
    """
    Validate a single metadata row against the core schema.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    for field in CORE_FIELDS:
        if field.name not in row:
            errors.append(f"Missing required field: {field.name}")
        elif row[field.name] is None and not field.nullable:
            errors.append(f"Null value in non-nullable field: {field.name}")

    if "split" in row and row["split"] not in VALID_SPLITS:
        errors.append(f"Invalid split '{row['split']}'. Must be one of {VALID_SPLITS}")

    if "label" in row and row["label"] is not None and row["label"] < 0:
        errors.append(f"Negative label index: {row['label']}")

    return errors


def _pa_to_pandas_dtype(pa_type):
    import pandas as pd
    mapping = {
        pa.string():  "object",
        pa.int32():   "Int32",
        pa.int64():   "Int64",
        pa.float32(): "float32",
        pa.float64(): "float64",
        pa.bool_():   "boolean",
    }
    return mapping.get(pa_type, "object")


def make_empty_dataframe():
    """Return an empty pandas DataFrame conforming to the full schema."""
    import pandas as pd
    cols = {f.name: pd.Series(dtype=_pa_to_pandas_dtype(f.type))
            for f in CORE_FIELDS + EXTENDED_FIELDS}
    return pd.DataFrame(cols)