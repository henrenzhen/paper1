import os
import re
import math
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

TEST_CSV = "/root/project/data/sim_test_parent_min3.csv"
VOCAB_CSV = "/root/project/data/rl_label_vocab.csv"
LOOKUP_CSV = "/root/project/data/attack_parent_lookup_for_llm.csv"

OUT_CSV = "/root/project/llm/llm_semantic_branch_smoke5_v6.csv"

PARENT_RE = re.compile(r"^T\d{4}$")


def parse_parent_ids(text):
    found = re.findall(r"T\d{4}", str(text))
    out = []
    seen = set()
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_prefix_semantic_from_lookup(prefix_ids_parent, lookup_map):
    ids = [x.strip() for x in str(prefix_ids_parent).split("||")]
    lines = []
    for i, tid in enumerate(ids):
        if not PARENT_RE.fullmatch(tid):
            continue
        info = lookup_map.get(tid, {"technique_name": tid, "tactic_titles": ""})
        name = info["technique_name"]
        tactics = info["tactic_titles"]
        if tactics and str(tactics).strip():
            lines.append(f"{i+1}. {tid} - {name} | tactics: {tactics}")
        else:
            lines.append(f"{i+1}. {tid} - {name}")
    return "\n".join(lines)


def logsumexp(xs):
    if not xs:
        return float("-inf")
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def softmax_from_scores(score_dict):
    keys = list(score_dict.keys())
    vals = [score_dict[k] for k in keys]
    lse = logsumexp(vals)
    return {k: math.exp(score_dict[k] - lse) for k in keys}


def compute_metrics(df):
    top1_hits = 0
    top5_hits = 0
    mrr_sum = 0.0

    for _, row in df.iterrows():
        gold = str(row["gold_next"]).strip()
        preds = [
            str(row["pred1"]).strip(),
            str(row["pred2"]).strip(),
            str(row["pred3"]).strip(),
            str(row["pred4"]).strip(),
            str(row["pred5"]).strip(),
        ]

        if gold == preds[0]:
            top1_hits += 1

        rr = 0.0
        for rank, pred in enumerate(preds, start=1):
            if gold == pred:
                top5_hits += 1
                rr = 1.0 / rank
                break
        mrr_sum += rr

    n = len(df)
    return {
        "top1": top1_hits / n if n else 0.0,
        "top5": top5_hits / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
    }


def build_yesno_messages(prefix_semantic, cand_id, cand_name, cand_tactics):
    system = (
        "You are a cybersecurity analyst specialized in MITRE ATT&CK. "
        "Judge whether a candidate parent technique is the immediate next step "
        "for the observed attack sequence."
    )

    if cand_tactics and str(cand_tactics).strip():
        cand_line = f"{cand_id} - {cand_name} | tactics: {cand_tactics}"
    else:
        cand_line = f"{cand_id} - {cand_name}"

    user = f"""Observed ATT&CK parent technique sequence:
{prefix_semantic}

Candidate next technique:
{cand_line}

Question:
Is the candidate the immediate next ATT&CK parent technique?

Strict rules:
1. Answer with only one word: Yes or No
2. Do not output any other text
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def request_yes_logprob(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    r = requests.post(API_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def extract_yes_logprob(resp_json):
    """
    Extract logprob for first generated token being 'Yes' if available.
    Fallback to actual generated token logprob if token == Yes.
    """
    yes_variants = {"Yes", " yes", "Yes.", "ĠYes", "▁Yes"}
    out = float("-inf")
    answer_text = ""

    try:
        choice0 = resp_json["choices"][0]
    except Exception:
        return -50.0, answer_text

    try:
        answer_text = choice0["message"]["content"]
    except Exception:
        answer_text = ""

    lp = choice0.get("logprobs", None)
    if not lp:
        return -50.0, answer_text

    content = lp.get("content", None)
    if isinstance(content, list) and len(content) > 0:
        tok0 = content[0]
        token = str(tok0.get("token", "")).strip()
        logprob = tok0.get("logprob", None)

        if token in {"Yes", "yes"} and logprob is not None:
            out = float(logprob)

        top_lps = tok0.get("top_logprobs", [])
        if isinstance(top_lps, list):
            for item in top_lps:
                t = str(item.get("token", ""))
                v = item.get("logprob", None)
                if t in yes_variants and v is not None:
                    out = max(out, float(v))

    if out == float("-inf"):
        out = -50.0

    return out, answer_text


def main():
    os.makedirs("/root/project/llm", exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    vocab_df = pd.read_csv(VOCAB_CSV)
    lookup_df = pd.read_csv(LOOKUP_CSV)

    valid_vocab_list = vocab_df["technique_id_parent"].astype(str).tolist()
    valid_vocab = set(valid_vocab_list)

    lookup_df = lookup_df[lookup_df["technique_id_parent"].astype(str).isin(valid_vocab)].copy()
    lookup_df["technique_id_parent"] = lookup_df["technique_id_parent"].astype(str)
    lookup_df["technique_name_parent"] = lookup_df["technique_name_parent"].astype(str)
    lookup_df["tactic_titles"] = lookup_df["tactic_titles"].fillna("").astype(str)

    lookup_map = {}
    task_space_items = []
    for _, row in lookup_df.iterrows():
        tid = row["technique_id_parent"]
        info = {
            "technique_id": tid,
            "technique_name": row["technique_name_parent"],
            "tactic_titles": row["tactic_titles"],
        }
        lookup_map[tid] = info
        task_space_items.append(info)

    task_space_items = [lookup_map[tid] for tid in valid_vocab_list if tid in lookup_map]

    # smoke5
    test_df = test_df.head(5).copy()
    rows = []

    for idx, row in test_df.iterrows():
        sequence_id = row["sequence_id"]
        prefix_len = int(row["prefix_len"])
        gold = str(row["next_technique_id_parent"]).strip()

        prefix_ids_parent = str(row["prefix_technique_ids_parent"])
        observed_set = set(parse_parent_ids(prefix_ids_parent))
        prefix_semantic = build_prefix_semantic_from_lookup(prefix_ids_parent, lookup_map)

        available_items = [x for x in task_space_items if x["technique_id"] not in observed_set]

        candidate_scores = {}
        raw_logs = []

        for j, item in enumerate(available_items):
            cand_id = item["technique_id"]
            cand_name = item["technique_name"]
            cand_tactics = item["tactic_titles"]

            messages = build_yesno_messages(
                prefix_semantic=prefix_semantic,
                cand_id=cand_id,
                cand_name=cand_name,
                cand_tactics=cand_tactics,
            )

            resp_json = request_yes_logprob(messages)
            yes_lp, answer_text = extract_yes_logprob(resp_json)

            candidate_scores[cand_id] = yes_lp

            raw_logs.append({
                "candidate_index": j + 1,
                "candidate_id": cand_id,
                "answer_text": answer_text,
                "yes_logprob": yes_lp,
            })

            if (j + 1) % 50 == 0:
                print(f"  scored {j+1}/{len(available_items)} candidates")

        probs_dict = softmax_from_scores(candidate_scores)

        ranked = sorted(
            [
                {
                    "technique_id": tid,
                    "score": candidate_scores[tid],
                    "prob": probs_dict[tid]
                }
                for tid in probs_dict
            ],
            key=lambda x: (-x["prob"], x["technique_id"])
        )

        top5 = ranked[:5]

        out = {
            "sequence_id": sequence_id,
            "prefix_len": prefix_len,
            "gold_next": gold,
            "prefix_ids_parent": prefix_ids_parent,
            "prefix_semantic": prefix_semantic,
            "yesno_raw": "\n".join([str(x) for x in raw_logs[:50]])  # 只保留前50条原始日志，避免CSV过大
        }

        for k in range(5):
            out[f"pred{k+1}"] = top5[k]["technique_id"]
            out[f"score{k+1}"] = top5[k]["score"]
            out[f"prob{k+1}"] = top5[k]["prob"]

        rows.append(out)

        print(f"[{idx+1:02d}/5] {sequence_id} prefix_len={prefix_len} gold={gold}")
        print("  top5    :", " || ".join([x["technique_id"] for x in top5]))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    metrics = compute_metrics(out_df)

    print("\nSaved to:", OUT_CSV)
    print("\n=== LLM Semantic Branch v6 Metrics (smoke5) ===")
    print(f"top1 = {metrics['top1']:.4f}")
    print(f"top5 = {metrics['top5']:.4f}")
    print(f"mrr  = {metrics['mrr']:.4f}")

    print("\nPreview:")
    print(out_df.head(5).to_string())


if __name__ == "__main__":
    main()