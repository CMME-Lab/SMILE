from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smile import data 
from smile.config import load, resolve_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="records JSON: a flat list, or a split file")
    p.add_argument("--dataset", default="d1", choices=["d1", "d2"])
    p.add_argument("--out", default=None, help="directory for the split file")
    p.add_argument("--from-split", action="store_true",
                   help="the input already carries train/val/test; validate and pass through")
    p.add_argument("--seed", type=int, default=None, help="override the config's split seed")
    p.add_argument("--check-only", action="store_true", help="report and exit without writing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = load(resolve_dataset("d1_public" if args.dataset == "d1" else "d2_internal"))
    spec = cfg["split"]
    view_fields = data.view_fields(cfg)
    reference = data.reference_field(cfg)

    print(f"dataset    {cfg['name']}")
    print(f"views      {view_fields}")

    if args.from_split:
        splits = data.load_split(args.records, path_fields=view_fields)
        print(f"source     {args.records}  (existing split, passed through)")
    else:
        records = data.load_records(args.records, path_fields=view_fields)
        ratios = {"train": float(spec.get("train", 0.0)),
                  "val": float(spec.get("val", 0.0)),
                  "test": float(spec.get("test", 0.0))}
        if not ratios["val"] and spec.get("val_fraction_of_train"):
            held = ratios["train"] * float(spec["val_fraction_of_train"])
            ratios["val"], ratios["train"] = held, ratios["train"] - held
        seed = args.seed if args.seed is not None else int(spec.get("seed", 42))
        splits = data.split_records(records, ratios=ratios, seed=seed,
                                    group_by=cfg["fields"]["patient_id"])
        print(f"source     {args.records}  ({len(records)} records)")
        print(f"split      fresh, ratios "
              f"{ {k: round(v, 4) for k, v in ratios.items()} }, seed {seed}")

    print(data.split_summary(splits, group_by=cfg["fields"]["patient_id"]))

    problems = data.check_split(splits, group_by=cfg["fields"]["patient_id"])
    if problems:
        print("problems found:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("no patient leakage, no empty captions")


    present = [f for f in view_fields if any(f in r for records in splits.values() for r in records)]
    if not present:
        found = sorted({k for records in splits.values() for r in records for k in r
                        if "view" in k.lower() or "path" in k.lower()})
        print(f"warning    missing view fields {view_fields}; found {found}")
    elif len(present) < len(view_fields):
        print(f"warning    only {present} of {view_fields} are present")

    unresolved = sum(
        1 for records in splits.values() for record in records
        for field in view_fields if record.get(field) and not Path(record[field]).exists()
    )
    if unresolved:
        print(f"warning    {unresolved} view paths do not resolve on this machine")

    missing_caption = sum(1 for records in splits.values() for r in records
                          if not str(r.get(reference) or "").strip())
    if missing_caption:
        print(f"warning    {missing_caption} records have no {reference!r}")

    if args.check_only:
        print("check only: nothing written")
        return 0 if not problems else 1


    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_split.json"
    out_path.write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote      {out_path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
