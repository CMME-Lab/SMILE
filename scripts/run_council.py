from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smile import data
from smile.agents import Agent
from smile.config import REPO_ROOT, load, load_agent, load_judge, resolve_dataset
from smile.council import Council, Judge

AGENT_NAMES = ("agent_a", "agent_b", "agent_c")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="input records JSON")
    p.add_argument("--dataset", default="d1", choices=["d1", "d2"])
    p.add_argument("--out", default=None, help="output JSON (default: runs/<dataset>_council.json)")
    p.add_argument("--stage", default="all", choices=["all", "agents", "judge"],
                   help="'all' runs agents then judge; 'judge' expects agent outputs present")
    p.add_argument("--judge-config", default=None,
                   help="default: configs/judge/medgemma_consensus.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any agent/judge config key; repeatable")
    p.add_argument("--limit", type=int, default=None, help="process only the first N records")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve config and print the first judge prompt; load nothing")
    return p.parse_args(argv)


def dataset_name(short: str) -> str:
    return "d1_public" if short == "d1" else "d2_internal"


def main(argv=None) -> int:
    args = parse_args(argv)

    dataset_cfg = load(resolve_dataset(dataset_name(args.dataset)))
    view_fields = data.view_fields(dataset_cfg)
    bank = dataset_cfg["question_bank"]

    records = data.load_records(args.records)
    if args.limit:
        records = records[: args.limit]

    judge_cfg = load_judge(
        args.judge_config or REPO_ROOT / "configs" / "judge" / "medgemma_consensus.yaml",
        overrides=args.overrides,
    )
    judge = Judge(judge_cfg)

    agents: dict[str, Agent] = {}
    if args.stage in ("all", "agents"):
        for name in AGENT_NAMES:
            cfg = load_agent(REPO_ROOT / "configs" / "agents" / f"{name}.yaml",
                             overrides=args.overrides)
            agents[name] = Agent(cfg)

    out_path = Path(args.out) if args.out else Path("runs") / f"{args.dataset}_council.json"

    print(f"records    {args.records}  ({len(records)} records)")
    print(f"dataset    {dataset_cfg['name']}  views={view_fields[:2]}  bank={bank}")
    print(f"stage      {args.stage}")
    print(f"judge      {judge_cfg.base_model}  (frozen, no adapter)")
    print(f"           reference prompts: {judge_cfg.reference_prompts.get('mode')}")
    for name, agent in agents.items():
        print(f"{name:<11}{agent.cfg.base_model}  adapter={agent.cfg.weights}")
    print(f"out        {out_path}")

    if args.dry_run:
        candidate_a = str(records[0].get("agent_A_caption") or records[0].get("output_A") or "")
        candidate_b = str(records[0].get("agent_B_report") or records[0].get("output_B") or "")
        text = judge.user_prompt(records[0], candidate_a=candidate_a, candidate_b=candidate_b)
        print("judge user prompt, first record")
        print(text)
        print(f"dry run: {len(text)} characters; nothing was loaded")
        return 0

    if args.stage in ("all", "agents"):
        for agent in agents.values():
            agent.load()
        records = Council(agents, judge).run_agents(
            records, view_fields, dataset=bank, progress=_progress("agents")
        )

    if args.stage in ("all", "judge"):
        judge.load()
        records = Council(agents, judge).run_judge(records, view_fields, progress=_progress("judge"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": records}, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")

    if args.stage in ("all", "judge"):
        failed = sum(1 for r in records
                     if str(r.get("judge_thoughts", {}).get("rules_check", {}).get("R1", ""))
                     .startswith("fail: invalid JSON"))
        empty = sum(1 for r in records if not str(r.get("judge_final_response") or "").strip())
        print(f"judge      {len(records) - failed}/{len(records)} returned parseable JSON")
        if empty:
            print(f"           {empty} empty captions (these score 0)")

    print(f"wrote      {out_path}")
    return 0


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"\r{label:<11}{done}/{total}", end="", flush=True)
    return report


if __name__ == "__main__":
    sys.exit(main())
