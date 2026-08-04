from pathlib import Path
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = PROJECT_ROOT / "rl" / "rl_baseline_v2.pt"
PARSED_ROOT = PROJECT_ROOT / "data_v2" / "external_ctid" / "parsed"
OUT_DIR = PROJECT_ROOT / "rl" / "all_ctid_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PLANS = [
    "apt29",
    "carbanak",
    "fin6",
    "fin7",
    "menu_pass",
    "oilrig",
    "sandworm",
    "wizard_spider",
    "turla_carbon",
    "turla_snake",
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize_state(s: str):
    return [x.strip() for x in str(s).split() if x.strip()]


def normalize_prefix(prefix_str: str) -> str:
    s = str(prefix_str).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return " ".join([str(x).strip() for x in arr if str(x).strip()])
        except Exception:
            pass
    items = [x.strip() for x in s.split("||") if x.strip()]
    return " ".join(items)


def encode_tokens(tokens, token2id, max_len):
    ids = [token2id.get(t, token2id[UNK_TOKEN]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [token2id[PAD_TOKEN]] * (max_len - len(ids))
    return ids


class CTIDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id, token2id, max_len):
        self.states = []
        self.labels = []
        self.meta = []

        for i, row in df.iterrows():
            state_str = normalize_prefix(row["prefix_technique_ids_parent"])
            tokens = tokenize_state(state_str)
            label_name = str(row["next_technique_id_parent"]).strip()

            if label_name not in label2id:
                continue

            self.states.append(torch.tensor(encode_tokens(tokens, token2id, max_len), dtype=torch.long))
            self.labels.append(torch.tensor(label2id[label_name], dtype=torch.long))
            self.meta.append({
                "sample_id": row.get("sample_id", f"ctid::{i}"),
                "org_name": row.get("org_name", ""),
                "plan_id": row.get("plan_id", ""),
                "scenario_id": row.get("scenario_id", ""),
                "prefix_len": int(row["prefix_len"]),
                "state": state_str,
                "true_label": label_name
            })

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx], self.meta[idx]


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas


class PolicyGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = h.squeeze(0)
        logits = self.classifier(h)
        return logits


@torch.no_grad()
def evaluate(model, loader, id2label):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    mrr_sum = 0.0
    prediction_rows = []

    for x, y, metas in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        ce_losses = criterion(logits, y)
        loss = ce_losses.mean()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        pred1 = probs.argmax(dim=1)
        top1 += (pred1 == y).sum().item()

        k = min(5, probs.size(1))
        top5_vals, top5_idx = torch.topk(probs, k=k, dim=1)
        top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()

        sorted_idx = torch.argsort(probs, dim=1, descending=True)
        for i in range(x.size(0)):
            true_label = y[i].item()
            ranking = sorted_idx[i].tolist()
            rank = ranking.index(true_label) + 1
            mrr_sum += 1.0 / rank

            pred_ids = top5_idx[i].tolist()
            pred_probs = top5_vals[i].tolist()
            pred_names = [id2label[idx] for idx in pred_ids]

            prediction_rows.append({
                "sample_id": metas[i]["sample_id"],
                "org_name": metas[i]["org_name"],
                "plan_id": metas[i]["plan_id"],
                "scenario_id": metas[i]["scenario_id"],
                "prefix_len": metas[i]["prefix_len"],
                "state": metas[i]["state"],
                "true_label": id2label[true_label],
                "true_rank": rank,
                "top1_pred": pred_names[0] if pred_names else "",
                "top1_prob": pred_probs[0] if pred_probs else "",
                "top5_labels": " || ".join(pred_names),
                "top5_probs": " || ".join([f"{p:.6f}" for p in pred_probs])
            })

    metrics = {
        "loss": total_loss / total if total > 0 else 0.0,
        "top1": top1 / total if total > 0 else 0.0,
        "top5": top5 / total if total > 0 else 0.0,
        "mrr": mrr_sum / total if total > 0 else 0.0,
        "samples": total,
    }
    return metrics, pd.DataFrame(prediction_rows)


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    token2id = ckpt["token2id"]
    label2id = ckpt["label2id"]
    id2label = {int(k): v for k, v in ckpt["id2label"].items()} if isinstance(next(iter(ckpt["id2label"].keys())), str) else ckpt["id2label"]
    num_labels = ckpt["num_labels"]
    max_len = ckpt["max_len"]

    model = PolicyGRU(
        vocab_size=len(token2id),
        emb_dim=128,
        hidden_dim=128,
        num_labels=num_labels,
        pad_idx=token2id[PAD_TOKEN]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    summary_rows = []

    for plan in TARGET_PLANS:
        in_path = PARSED_ROOT / f"ctid_eval_parent_{plan}_in184.csv"
        if not in_path.exists():
            print(f"[SKIP] missing: {in_path}")
            continue

        df = pd.read_csv(in_path, encoding="utf-8-sig")
        ds = CTIDDataset(df, label2id, token2id, max_len)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        metrics, pred_df = evaluate(model, loader, id2label)

        pred_out = OUT_DIR / f"rl_{plan}_predictions_top5.csv"
        pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "plan_name": plan,
            "samples": metrics["samples"],
            "loss": round(metrics["loss"], 4),
            "top1": round(metrics["top1"], 4),
            "top5": round(metrics["top5"], 4),
            "mrr": round(metrics["mrr"], 4),
            "predictions_file": str(pred_out),
        })

        print(f"[OK] {plan}: samples={metrics['samples']} top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} mrr={metrics['mrr']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_out = OUT_DIR / "rl_all_ctid_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("\n=== RL ALL-PLAN SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved -> {summary_out}")


if __name__ == "__main__":
    main()