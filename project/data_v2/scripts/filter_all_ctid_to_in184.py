from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSED_ROOT = PROJECT_ROOT / "data_v2" / "external_ctid" / "parsed"
LABEL_PATH = PROJECT_ROOT / "data" / "rl_label_vocab.csv"

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


def main():
    labels = pd.read_csv(LABEL_PATH, encoding="utf-8-sig")
    label_set = set(labels["technique_id_parent"].astype(str).str.strip())

    summary_rows = []

    for plan in TARGET_PLANS:
        in_path = PARSED_ROOT / f"ctid_eval_parent_{plan}.csv"
        out_path = PARSED_ROOT / f"ctid_eval_parent_{plan}_in184.csv"

        if not in_path.exists():
            summary_rows.append({
                "plan_name": plan,
                "status": "missing_input",
                "original_samples": 0,
                "kept_samples": 0,
                "dropped_samples": 0,
                "coverage_ratio": 0.0,
                "dropped_labels": "",
            })
            continue

        df = pd.read_csv(in_path, encoding="utf-8-sig")
        df["next_technique_id_parent"] = df["next_technique_id_parent"].astype(str).str.strip()

        keep_df = df[df["next_technique_id_parent"].isin(label_set)].copy()
        drop_df = df[~df["next_technique_id_parent"].isin(label_set)].copy()

        keep_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        original = len(df)
        kept = len(keep_df)
        dropped = len(drop_df)
        coverage = kept / original if original > 0 else 0.0
        dropped_labels = sorted(drop_df["next_technique_id_parent"].unique().tolist()) if dropped > 0 else []

        summary_rows.append({
            "plan_name": plan,
            "status": "ok",
            "original_samples": original,
            "kept_samples": kept,
            "dropped_samples": dropped,
            "coverage_ratio": round(coverage, 4),
            "dropped_labels": " || ".join(dropped_labels),
        })

        print(f"[OK] {plan}: kept={kept}/{original}, dropped={dropped}")

    summary_df = pd.DataFrame(summary_rows)
    summary_out = PARSED_ROOT / "ctid_in184_filter_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("\n=== FILTER SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved -> {summary_out}")


if __name__ == "__main__":
    main()