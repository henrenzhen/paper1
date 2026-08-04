from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CKPT_PATH = BASE_DIR / "rl_baseline_v2.pt"
INPUT_FILES = [
    PROJECT_ROOT / "data_v2" / "micro_state" / "apt29_aligned_richer_input.csv",
    PROJECT_ROOT / "data_v2" / "micro_state" / "oilrig_aligned_richer_input.csv",
    PROJECT_ROOT / "data_v2" / "micro_state" / "sandworm_aligned_richer_input.csv",
    PROJECT_ROOT / "data_v2" / "micro_state" / "turla_carbon_aligned_richer_input.csv",
]
OUTPUT_CSV = PROJECT_ROOT / "data_v2" / "micro_state" / "rl_on_aligned_56_predictions_top5.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def tokenize_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]


def encode_tokens(tokens, token2id, max_len):
    ids = [token2id.get(t, token2id.get("<UNK>", 1)) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [token2id.get("<PAD>", 0)] * (max_len - len(ids))
    return ids


class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = h.squeeze(0)
        logits = self.classifier(h)
        return logits


class InferenceDataset(Dataset):
    def __init__(self, df, token2id, max_len):
        self.rows = []
        self.states = []

        for _, row in df.iterrows():
            state_str = str(row["state"]).strip()
            tokens = tokenize_state(state_str)
            ids = encode_tokens(tokens, token2id, max_len)

            self.states.append(torch.tensor(ids, dtype=torch.long))
            self.rows.append({
                "annotation_id": str(row["annotation_id"]).strip(),
                "source_org": str(row["source_org"]).strip(),
                "state": state_str,
                "true_label": str(row["true_label"]).strip(),
            })

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.rows[idx]


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return xs, metas


@torch.no_grad()
def run_inference(model, loader, id2label):
    model.eval()
    rows = []

    for x, metas in loader:
        x = x.to(DEVICE)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        k = min(5, probs.size(1))
        top5_vals, top5_idx = torch.topk(probs, k=k, dim=1)
        sorted_idx = torch.argsort(probs, dim=1, descending=True)

        for i in range(x.size(0)):
            true_label = metas[i]["true_label"]
            ranking = sorted_idx[i].tolist()
            ranked_names = [id2label[idx] for idx in ranking]

            true_rank = ranked_names.index(true_label) + 1 if true_label in ranked_names else -1

            pred_ids = top5_idx[i].tolist()
            pred_probs = top5_vals[i].tolist()
            pred_names = [id2label[idx] for idx in pred_ids]

            rows.append({
                "annotation_id": metas[i]["annotation_id"],
                "source_org": metas[i]["source_org"],
                "state": metas[i]["state"],
                "true_label": true_label,
                "true_rank": true_rank,
                "top1_pred": pred_names[0] if pred_names else "",
                "top1_prob": pred_probs[0] if pred_probs else "",
                "top5_labels": " || ".join(pred_names),
                "top5_probs": " || ".join([f"{p:.6f}" for p in pred_probs]),
            })

    return pd.DataFrame(rows)


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

    frames = []
    for p in INPUT_FILES:
        if not p.exists():
            print(f"[WARN] Missing input, skipped: {p}")
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "alignment_ok" in df.columns:
            df = df[df["alignment_ok"] == 1].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No aligned input files found.")

    df = pd.concat(frames, ignore_index=True)
    df["annotation_id"] = df["annotation_id"].astype(str).str.strip()
    df["source_org"] = df["source_org"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    df["true_label"] = df["true_label"].astype(str).str.strip()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    token2id = ckpt["token2id"]
    id2label = ckpt["id2label"]
    num_labels = ckpt["num_labels"]
    max_len = ckpt["max_len"]

    model = PolicyGRU(
        vocab_size=len(token2id),
        emb_dim=128,
        hidden_dim=128,
        num_labels=num_labels,
        pad_idx=token2id.get("<PAD>", 0),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = InferenceDataset(df, token2id, max_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    out_df = run_inference(model, loader, id2label)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Checkpoint : {CKPT_PATH}")
    print(f"[INFO] Input rows : {len(df)}")
    print(f"[INFO] Output rows: {len(out_df)}")
    print(f"[INFO] Saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()