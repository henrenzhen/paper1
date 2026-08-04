import json
import re
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


INPUT_CSV = DATA_DIR / "sim_val_parent_min3_kg_context.csv"
OUTPUT_CSV = DATA_DIR / "sim_val_llm_cot.csv"


MAX_WORKERS = 30

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "/model-storage/model/Qwen3.5-35B-A3B-FP8"

def normalize_str(x):
    return str(x).strip() if x is not None else ""

def truncate_text(text: str, max_chars: int = 700):
    text = normalize_str(text)
    if len(text) <= max_chars: return text
    clipped = text[:max_chars]
    if " " in clipped: clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip() + " ..."

def build_system_prompt():
    return """
你是一个高级 APT 威胁狩猎专家与 ATT&CK 攻击图分析师。
你的任务是：基于攻击者已执行的 ATT&CK 技术序列（Prefix）以及相关的知识图谱上下文（KG Context），推断攻击者当前的【阶段性操作状态】，并直接预测下一步最可能执行的 5 个 ATT&CK Parent Technique（父技术）。

由于输入数据是模拟的宏观序列，你必须遵守以下严格限制：
1. 绝对不要凭空捏造微观动作。推理必须完全基于传入的 Prefix ID 及 KG Context 进行逻辑推演。
2. 预测结果必须是纯粹的父技术 ID。

请在 JSON 的 `_thinking_process` 字段中写下你的推理过程，按以下三步进行思考：
[战术阶段评估]：分析 Prefix 中最后两步，它们处于什么战术阶段？
[已获资产推演]：基于前缀技术，攻击者目前掌握了什么级别的粗粒度资产或权限？
[意图图谱映射]：结合 KG Context，前缀的最后几步操作最可能为后续攻击开启了什么逻辑攻击面？

推理完成后，请在 `predicted_next_ttps` 数组中输出恰好 5 个最可能的下一步 ATT&CK 父技术 ID。
""".strip()

def build_user_prompt(row):
    prefix = normalize_str(row.get("prefix_technique_ids_parent", ""))
    recent_ids = normalize_str(row.get("recent_prefix_ids", ""))
    kg_context = truncate_text(row.get("kg_context_text", ""), max_chars=700)
    return f"### 攻击前缀序列 (Prefix) ###\n{prefix}\n(重点关注最后两步：{recent_ids})\n\n### 相关的知识图谱上下文 (KG Context) ###\n{kg_context}\n\n### 任务要求 ###\n请先在 `_thinking_process` 字段推演，随后在 `predicted_next_ttps` 数组输出 5 个预测的父技术 ID。"

client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

def process_row(row_tuple):
    index, row = row_tuple
    user_prompt = build_user_prompt(row)
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_direct_prediction",
            "schema": {
                "type": "object",
                "properties": {
                    "_thinking_process": {"type": "string"},
                    "predicted_next_ttps": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["_thinking_process", "predicted_next_ttps"],
                "additionalProperties": False,
            }
        }
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_text = normalize_str(resp.choices[0].message.content)
        
        parsed = {}
        try:
            parsed = json.loads(raw_text, strict=False)
        except:
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                try: parsed = json.loads(match.group(1), strict=False)
                except: pass

        return {
            "sequence_id": row.get("sequence_id", ""),
            "state": row.get("prefix_technique_ids_parent", ""),
            "true_label": row.get("next_technique_id_parent", ""),
            "llm_thinking_process": normalize_str(parsed.get("_thinking_process", "")),
            "predicted_next_ttps": json.dumps(parsed.get("predicted_next_ttps", [])),
            "raw_output": raw_text
        }
    except Exception as e:
        return {
            "sequence_id": row.get("sequence_id", ""),
            "state": row.get("prefix_technique_ids_parent", ""),
            "true_label": row.get("next_technique_id_parent", ""),
            "llm_thinking_process": f"[ERROR] {e}",
            "predicted_next_ttps": "[]",
            "raw_output": ""
        }

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] 找不到输入文件: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    total = len(df)
    
    print(f"[INFO] 开始多线程并发生成推演文本，总数据量: {total}")
    print(f"[INFO] 线程池大小: {MAX_WORKERS}")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row_tuple): row_tuple for row_tuple in df.iterrows()}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"\r进度: {completed}/{total} ({(completed/total)*100:.1f}%)", end="", flush=True)

    print("\n[INFO] 处理完成，正在保存结果...")
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] 结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()