from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from smile import data
from smile.config import load, resolve_dataset
from smile.eval import clinical_efficacy as ce
from smile.eval import nlg, vqa

FAMILIES = ("nlg", "ce", "vqa")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="results JSON from run_council.py")
    p.add_argument(
        "--dataset",
        default="d1",
        choices=["d1", "d2"],
        help="which question bank / CE policy",
    )
    p.add_argument(
        "--pred-col", required=True, help="column holding the generated text"
    )
    p.add_argument(
        "--gt-col",
        default=None,
        help="reference column (default: the dataset config's)",
    )
    p.add_argument("--metrics", default=None)
    p.add_argument(
        "--out", default=None, help="directory for per-entity / per-question CSVs"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    requested = (
        [m.strip() for m in args.metrics.split(",")] if args.metrics else list(FAMILIES)
    )
    families = [m for m in requested if m in FAMILIES]
    nlg_keys = [m for m in requested if m in nlg.METRIC_KEYS]
    if nlg_keys and "nlg" not in families:
        families.append("nlg")

    dataset_cfg = load(
        resolve_dataset("d1_public" if args.dataset == "d1" else "d2_internal")
    )
    ce_policy = dataset_cfg.get("clinical_efficacy") or {}
    pred_col = args.pred_col
    gt_col = args.gt_col or data.reference_field(dataset_cfg)

    records = data.load_records(args.results)
    print(f"results    {args.results}  ({len(records)} records)")
    print(f"dataset    {dataset_cfg['name']}")
    print(f"scoring    {pred_col}  against  {gt_col}")

    references = [str(r.get(gt_col) or "") for r in records]
    predictions = [str(r.get(pred_col) or "") for r in records]
    empty = sum(1 for p in predictions if not p.strip())
    if empty:
        print(
            f"warning    {empty}/{len(predictions)} predictions are empty and score 0"
        )

    results: dict[str, object] = {
        "results": args.results,
        "pred_col": pred_col,
        "gt_col": gt_col,
        "n_records": len(records),
    }
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if "nlg" in families:
        scores = nlg.compute(
            references, predictions, metrics=nlg_keys or nlg.METRIC_KEYS
        )
        results["nlg"] = scores
        print("NLG")
        for key, value in scores.items():
            print(f"  {key:<12}{value:.4f}")

    if "ce" in families:
        if not ce_policy.get("enabled", True):
            print(f"CE         not enabled for {dataset_cfg['name']}; skipping")
        else:
            scores = ce.compute(references, predictions)
            results["ce"] = {k: scores[k] for k in ("micro", "macro")}
            entities = scores["per_entity"]
            print(f"Clinical efficacy  ({len(entities)} canonical entities)")
            for name in ("micro", "macro"):
                row = scores[name]
                print(
                    f"  {name:<6} acc {row['accuracy']:.4f}  prec {row['precision']:.4f}  "
                    f"rec {row['recall']:.4f}  f1 {row['f1-score']:.4f}"
                )
            if out_dir:
                _write_csv(out_dir / "clinical_metrics.csv", entities)

    if "vqa" in families:
        scores = vqa.compute(records)
        if not scores["n"]:
            print("VQA        no scorable question/answer pairs in this file")
        else:
            results["vqa"] = scores
            print(
                f"VQA        n={scores['n']}, {scores['excluded']} excluded as not assessable"
            )
            print(
                f"  acc {scores['accuracy']:.4f}  prec {scores['precision']:.4f}  "
                f"rec {scores['recall']:.4f}  f1 {scores['f1-score']:.4f}"
            )

    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"wrote      {out_dir}/summary.json")

    return 0


def _write_csv(path: Path, rows: dict[str, dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "accuracy", "precision", "recall", "f1-score"])
        for name, row in rows.items():
            writer.writerow(
                [
                    name,
                    row["accuracy"],
                    row["precision"],
                    row["recall"],
                    row["f1-score"],
                ]
            )


if __name__ == "__main__":
    sys.exit(main())
