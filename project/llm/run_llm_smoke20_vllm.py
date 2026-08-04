import os
import re
import json
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

TEST_CSV = "/root/project/data/sim_test_parent_min3.csv"
RL_CSV = "/root/project/rl/rl_v2_test_predictions_top5.csv"
VOCAB_CSV = "/root/project/data/rl_label_vocab.csv"

OUT_CSV = "/root/project/llm/llm_test_predictions_top5_smoke20.csv"

PARENT_RE = re.compile(r"^T\d{4}$")


def normalize_scores(items):
    vals = [max(float(x["score"]), 0.0) for x in items]
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(items)] * len(items)
    return [v / s for v in vals]


def build_prompt(seq_text):
    return f"""Observed ATT&CK parent technique sequence:
{seq_text}

Task:
Predict the 5 most likely NEXT ATT&CK parent technique IDs.

Strict rules:
1. Only output parent ATT&CK technique IDs matching regex ^T[0-9]{{4}}$
2. Do not output any sub-technique like T1059.001
3. Do not repeat any technique already present in the observed sequence
4. Return exactly 5 items
5. Output exactly one JSON object with key "top5"
6. Each item in top5 must contain:
   - "technique_id": parent ATT&CK ID
   - "score": numeric score
7. No explanation, no extra text
"""


def call_vllm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a cybersecurity analyst. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def parse_observed_set(seq_text):
    parts = [x.strip() for x in str(seq_text).split("->")]
    return set([x for x in parts if PARENT_RE.fullmatch(x)])


def clean_top5(raw_obj, observed_set, label_vocab):
    if not raw_obj or "top5" not in raw_obj or not isinstance(raw_obj["top5"], list):
        return []

    cleaned = []
    seen = set()

    for item in raw_obj["top5"]:
        if not isinstance(item, dict):
            continue

        tid = str(item.get("technique_id", "")).strip()
        score = item.get("score", 0.0)

        if not PARENT_RE.fullmatch(tid):
            continue
        if tid in observed_set:
            continue
        if tid not in label_vocab:
            continue
        if tid in seen:
            continue

        try:
            score = float(score)
        except Exception:
            score = 0.0

        cleaned.append({
            "technique_id": tid,
            "score": score
        })
        seen.add(tid)

    return cleaned[:5]


def backfill_to_top5(items, observed_set, label_vocab):
    used = {x["technique_id"] for x in items}
    for tid in label_vocab:
        if tid in observed_set:
            continue
        if tid in used:
            continue
        items.append({"technique_id": tid, "score": 0.0})
        used.add(tid)
        if len(items) >= 5:
            break
    return items[:5]


def main():
    os.makedirs("/root/project/llm", exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    rl_df = pd.read_csv(RL_CSV)
    vocab_df = pd.read_csv(VOCAB_CSV)

    label_vocab = vocab_df["technique_id_parent"].astype(str).tolist()

    # 只取前20条 smoke test
    test_df = test_df.head(20).copy()

    # 构造 LLM 输入序列：把 " || " 转成 " -> "
    test_df["llm_input_sequence"] = (
        test_df["prefix_technique_ids_parent"]
        .astype(str)
        .str.replace(r"\s*\|\|\s*", " -> ", regex=True)
    )

    # 与 RL 对齐检查
    merged = test_df.merge(
        rl_df,
        on=["sequence_id", "prefix_len"],
        how="left",
        validate="one_to_one"
    )

    print("smoke rows =", len(merged))
    print("rows with matched RL =", merged["top5_labels"].notna().sum())

    rows = []

    for idx, row in merged.iterrows():
        sequence_id = row["sequence_id"]
        prefix_len = int(row["prefix_len"])
        seq_text = str(row["llm_input_sequence"])
        gold = str(row["next_technique_id_parent"])
        observed_set = parse_observed_set(seq_text)

        raw_text = ""
        err = ""

        try:
            prompt = build_prompt(seq_text)
            raw_text = call_vllm(prompt)
            raw_obj = extract_json(raw_text)

            preds = clean_top5(raw_obj, observed_set, label_vocab)
            preds = backfill_to_top5(preds, observed_set, label_vocab)
            probs = normalize_scores(preds)

        except Exception as e:
            err = repr(e)
            raw_text = f"ERROR: {err}"
            preds = [{"technique_id": tid, "score": 0.0} for tid in label_vocab[:5]]
            probs = [0.2] * 5

        out = {
            "sequence_id": sequence_id,
            "prefix_len": prefix_len,
            "input_sequence": seq_text,
            "gold_next": gold,
            "raw_response": raw_text,
            "error": err,
            "rl_top1_pred": row.get("top1_pred", ""),
            "rl_top1_prob": row.get("top1_prob", ""),
            "rl_top5_labels": row.get("top5_labels", ""),
            "rl_top5_probs": row.get("top5_probs", "")
        }

        for k in range(5):
            out[f"pred{k+1}"] = preds[k]["technique_id"]
            out[f"score{k+1}"] = preds[k]["score"]
            out[f"prob{k+1}"] = probs[k]

        rows.append(out)

        print(f"[{idx+1:02d}/20] {sequence_id} prefix_len={prefix_len} gold={gold}")
        print("  input :", seq_text)
        print("  pred  :", " || ".join([x["technique_id"] for x in preds]))
        if err:
            print("  error :", err)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print("\nSaved to:", OUT_CSV)
    print(out_df.head(3).to_string())


if __name__ == "__main__":
    main()