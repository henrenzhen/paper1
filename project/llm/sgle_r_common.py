from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


# =========================
# Paths
# =========================

PROJECT_ROOT = Path("/root/project")
DATA_DIR = PROJECT_ROOT / "data"


# =========================
# MITRE tactic enum
# Stage 1 predicted_tactics must be normalized into this closed set.
# =========================

MITRE_TACTIC_ENUM: List[str] = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]

# Common aliases / spelling variants -> canonical enum
TACTIC_ALIAS_MAP: Dict[str, str] = {
    "resource development": "resource-development",
    "resource-development": "resource-development",
    "initial access": "initial-access",
    "initial-access": "initial-access",
    "privilege escalation": "privilege-escalation",
    "privilege-escalation": "privilege-escalation",
    "defense evasion": "defense-evasion",
    "defense-evasion": "defense-evasion",
    "credential access": "credential-access",
    "credential-access": "credential-access",
    "lateral movement": "lateral-movement",
    "lateral-movement": "lateral-movement",
    "command and control": "command-and-control",
    "command-and-control": "command-and-control",
    "command & control": "command-and-control",
    "c2": "command-and-control",
}


# =========================
# Data containers
# =========================

@dataclass
class LabelVocabItem:
    label_idx: int
    parent_tid: str
    parent_name: Optional[str] = None


# =========================
# Basic loaders
# =========================

def load_split_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load train/val/test split CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Split CSV is empty: {csv_path}")
    return df


def load_label_vocab(csv_path: str | Path = DATA_DIR / "rl_label_vocab.csv") -> pd.DataFrame:
    """
    Load parent-tech label vocab.

    Actual schema in this project:
    - label_id
    - technique_id_parent
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Label vocab CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Label vocab CSV is empty: {csv_path}")

    return df


def load_attack_parent_lookup(
    csv_path: str | Path = DATA_DIR / "attack_parent_lookup_for_llm.csv",
) -> pd.DataFrame:
    """
    Load lookup table used for LLM semantic context construction.

    Actual schema in this project includes:
    - technique_id_parent
    - technique_name_parent
    - tactic_ids
    - tactic_titles
    - member_technique_ids
    - member_technique_names
    - subtechnique_ids
    - subtechnique_names
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Attack parent lookup CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Attack parent lookup CSV is empty: {csv_path}")
    return df


# =========================
# Column inference helpers
# =========================

def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def infer_label_vocab_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer key columns from rl_label_vocab.csv.
    """
    label_idx_col = _find_first_existing_column(
        df, ["label_idx", "label_id", "class_idx", "y", "idx"]
    )
    parent_tid_col = _find_first_existing_column(
        df,
        [
            "technique_id_parent",
            "parent_tid",
            "tid",
            "technique_id",
            "label",
            "parent_technique_id",
        ],
    )

    parent_name_col = None
    for c in ["parent_name", "technique_name_parent", "technique_name", "name", "label_name"]:
        if c in df.columns:
            parent_name_col = c
            break

    return {
        "label_idx": label_idx_col,
        "parent_tid": parent_tid_col,
        "parent_name": parent_name_col,
    }


def infer_lookup_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer useful columns from attack_parent_lookup_for_llm.csv
    using the actual project schema.
    """
    cols: Dict[str, Optional[str]] = {}

    cols["parent_tid"] = _find_first_existing_column(
        df,
        [
            "technique_id_parent",
            "parent_tid",
            "tid",
            "technique_id",
            "parent_technique_id",
        ],
    )

    for key, candidates in {
        "parent_name": ["technique_name_parent", "parent_name", "technique_name", "name"],
        "description": ["description", "summary", "technique_description"],
        "tactics": ["tactic_ids", "tactic_titles", "tactics", "tactic", "mitre_tactics"],
        "detection": ["detection", "detect", "detection_notes"],
        "examples": ["examples", "procedure_examples", "procedures"],
        "member_technique_ids": ["member_technique_ids"],
        "member_technique_names": ["member_technique_names"],
        "subtechnique_ids": ["subtechnique_ids"],
        "subtechnique_names": ["subtechnique_names"],
    }.items():
        cols[key] = next((c for c in candidates if c in df.columns), None)

    return cols


# =========================
# Tactic normalization / validation
# =========================

def normalize_text_basic(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_tactic_name(tactic: str) -> Optional[str]:
    """
    Normalize a tactic string into canonical MITRE_TACTIC_ENUM entry.
    Returns None if the input cannot be mapped.
    """
    if tactic is None:
        return None

    x = normalize_text_basic(tactic)
    x = x.replace(" - ", "-")

    if x in MITRE_TACTIC_ENUM:
        return x

    if x in TACTIC_ALIAS_MAP:
        return TACTIC_ALIAS_MAP[x]

    x_space = x.replace("-", " ")
    if x_space in TACTIC_ALIAS_MAP:
        return TACTIC_ALIAS_MAP[x_space]

    x_dash = x.replace(" ", "-")
    if x_dash in MITRE_TACTIC_ENUM:
        return x_dash

    return None


def normalize_tactic_list(tactics: Sequence[str]) -> List[str]:
    """
    Normalize a tactic list, drop invalid items, de-duplicate while preserving order.
    """
    out: List[str] = []
    seen = set()

    for t in tactics:
        norm = normalize_tactic_name(t)
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)

    return out


def are_all_tactics_valid(tactics: Sequence[str]) -> bool:
    return all(t in MITRE_TACTIC_ENUM for t in tactics)


def parse_lookup_tactic_ids(raw_value: object) -> List[str]:
    """
    Parse tactic_ids field such as:
    'command-and-control'
    or 'execution || persistence'
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []

    parts = [str(x).strip() for x in str(raw_value).split("||")]
    return normalize_tactic_list([p for p in parts if p])


# =========================
# Label vocab utilities
# =========================

def build_label_idx_to_tid(label_vocab_df: pd.DataFrame) -> Dict[int, str]:
    cols = infer_label_vocab_columns(label_vocab_df)
    return {
        int(row[cols["label_idx"]]): str(row[cols["parent_tid"]]).strip()
        for _, row in label_vocab_df.iterrows()
    }


def build_tid_to_label_idx(label_vocab_df: pd.DataFrame) -> Dict[str, int]:
    cols = infer_label_vocab_columns(label_vocab_df)
    return {
        str(row[cols["parent_tid"]]).strip(): int(row[cols["label_idx"]])
        for _, row in label_vocab_df.iterrows()
    }


def get_all_parent_tids(label_vocab_df: pd.DataFrame) -> List[str]:
    cols = infer_label_vocab_columns(label_vocab_df)
    tids = label_vocab_df[cols["parent_tid"]].astype(str).str.strip().tolist()
    return tids


# =========================
# Semantic context helper
# =========================

def _clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_parent_semantic_context(
    parent_tids: Sequence[str],
    lookup_df: pd.DataFrame,
) -> str:
    """
    Build a compact semantic narrative block from attack_parent_lookup_for_llm.csv.
    """
    if not parent_tids:
        return "No prior parent techniques available."

    cols = infer_lookup_columns(lookup_df)
    tid_col = cols["parent_tid"]
    assert tid_col is not None

    wanted = {str(t).strip() for t in parent_tids}
    sub = lookup_df[lookup_df[tid_col].astype(str).str.strip().isin(wanted)].copy()

    rows: List[str] = []
    for _, row in sub.iterrows():
        tid = _clean_text(row[tid_col])
        pieces = [tid]

        if cols["parent_name"] is not None:
            parent_name = _clean_text(row[cols["parent_name"]])
            if parent_name:
                pieces.append(f"name={parent_name}")

        if cols["tactics"] is not None:
            tactic_ids = _clean_text(row[cols["tactics"]])
            if tactic_ids:
                pieces.append(f"tactics={tactic_ids}")

        if cols["member_technique_names"] is not None:
            member_names = _clean_text(row[cols["member_technique_names"]])
            if member_names:
                pieces.append(f"members={member_names}")

        if cols["subtechnique_names"] is not None:
            sub_names = _clean_text(row[cols["subtechnique_names"]])
            if sub_names:
                pieces.append(f"subtechniques={sub_names}")

        if cols["description"] is not None:
            desc = _clean_text(row[cols["description"]])
            if desc:
                pieces.append(f"description={desc}")

        rows.append(" | ".join(pieces))

    if not rows:
        return "No semantic lookup entries matched the prior parent techniques."

    return "\n".join(f"- {r}" for r in rows)


# =========================
# Metrics
# =========================

def compute_topk_accuracy(
    y_true: Sequence[str],
    ranked_preds: Sequence[Sequence[str]],
    k: int,
) -> float:
    if len(y_true) != len(ranked_preds):
        raise ValueError("y_true and ranked_preds must have the same length")
    if len(y_true) == 0:
        return 0.0

    hits = 0
    for gold, preds in zip(y_true, ranked_preds):
        if gold in list(preds)[:k]:
            hits += 1
    return hits / len(y_true)


def compute_mrr(
    y_true: Sequence[str],
    ranked_preds: Sequence[Sequence[str]],
) -> float:
    if len(y_true) != len(ranked_preds):
        raise ValueError("y_true and ranked_preds must have the same length")
    if len(y_true) == 0:
        return 0.0

    rr_sum = 0.0
    for gold, preds in zip(y_true, ranked_preds):
        rank = 0
        for i, p in enumerate(preds, start=1):
            if p == gold:
                rank = i
                break
        rr_sum += 0.0 if rank == 0 else 1.0 / rank

    return rr_sum / len(y_true)


def compute_basic_metrics(
    y_true: Sequence[str],
    ranked_preds: Sequence[Sequence[str]],
) -> Dict[str, float]:
    return {
        "top1": compute_topk_accuracy(y_true, ranked_preds, k=1),
        "top5": compute_topk_accuracy(y_true, ranked_preds, k=5),
        "mrr": compute_mrr(y_true, ranked_preds),
    }


# =========================
# Score / ranking helpers
# =========================

def softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if len(xs) == 0:
        return []

    xs_scaled = [x / temperature for x in xs]
    m = max(xs_scaled)
    exps = [math.exp(x - m) for x in xs_scaled]
    z = sum(exps)
    return [e / z for e in exps]


def rank_labels_from_score_dict(score_dict: Dict[str, float], descending: bool = True) -> List[str]:
    return [
        label
        for label, _ in sorted(
            score_dict.items(),
            key=lambda kv: kv[1],
            reverse=descending,
        )
    ]


# =========================
# Minimal self-check
# =========================

if __name__ == "__main__":
    vocab_df = load_label_vocab()
    lookup_df = load_attack_parent_lookup()

    print("label_vocab shape:", vocab_df.shape)
    print("lookup shape:", lookup_df.shape)
    print("label_vocab columns:", infer_label_vocab_columns(vocab_df))
    print("lookup columns:", infer_lookup_columns(lookup_df))
    print("num canonical tactics:", len(MITRE_TACTIC_ENUM))
    print("first 5 parent tids:", get_all_parent_tids(vocab_df)[:5])

    demo_tactics = parse_lookup_tactic_ids(lookup_df.iloc[0]["tactic_ids"])
    print("demo parsed tactics:", demo_tactics)

    demo_context = build_parent_semantic_context(
        parent_tids=get_all_parent_tids(vocab_df)[:3],
        lookup_df=lookup_df,
    )
    print("demo semantic context:")
    print(demo_context)