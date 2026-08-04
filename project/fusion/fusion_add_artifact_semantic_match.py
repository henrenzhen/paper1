from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path("/root/project")
FUSION_DIR = PROJECT_ROOT / "fusion"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    all_vecs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.cpu().numpy())

    return np.vstack(all_vecs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_features_top5.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(FUSION_DIR / "sgle_r_fusion_features_top5_semantic.csv"),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="basel/ATTACK-BERT",
    )
    args = parser.parse_args()

    ensure_dir(Path(args.output_csv).parent)

    df = pd.read_csv(args.input_csv).copy()
    required = ["instance_id", "artifact_query_text", "candidate_semantic_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, use_safetensors=True).to(device)

    query_texts = df["artifact_query_text"].fillna("").astype(str).tolist()
    cand_texts = df["candidate_semantic_text"].fillna("").astype(str).tolist()

    query_emb = encode_texts(query_texts, tokenizer, model, device)
    cand_emb = encode_texts(cand_texts, tokenizer, model, device)

    sims = np.sum(query_emb * cand_emb, axis=1)
    df["artifact_semantic_match"] = sims

    rank_col = np.zeros(len(df), dtype=int)
    gap_map = {}

    for instance_id, g in df.groupby("instance_id", sort=False):
        idx = g.index.to_numpy()
        vals = g["artifact_semantic_match"].to_numpy()
        order = np.argsort(-vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(vals) + 1)
        rank_col[idx] = ranks

        sorted_vals = np.sort(vals)[::-1]
        gap = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) >= 2 else 0.0
        gap_map[instance_id] = gap

    df["artifact_match_rank"] = rank_col
    df["artifact_match_gap_top1_top2"] = df["instance_id"].map(gap_map)

    df.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print(
        df[
            [
                "instance_id",
                "candidate_tid",
                "artifact_query_text",
                "candidate_semantic_text",
                "artifact_semantic_match",
                "artifact_match_rank",
                "artifact_match_gap_top1_top2",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()