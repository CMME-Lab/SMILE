from __future__ import annotations

import argparse
import sys

from datasets import Dataset
from trl import SFTTrainer

from smile import data
from smile.agents import Agent
from smile.config import load, load_agent, resolve_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="an agent config, e.g. configs/agents/agent_a.yaml")
    p.add_argument("--records", required=True, help="training records JSON")
    p.add_argument("--output-dir", default=None, help="where to write the adapter")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key; repeatable")
    p.add_argument("--dataset-config", default=None,
                   help="defaults to configs/datasets/<agent's dataset>.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve config, build the dataset and collator, then stop")
    return p.parse_args(argv)


def build(args):
    cfg = load_agent(args.config, overrides=args.overrides)

    dataset_path = args.dataset_config or resolve_dataset(cfg.dataset)
    dataset_cfg = load(dataset_path)

    records = data.load_records(args.records)
    fields = data.view_fields(dataset_cfg)
    reference = data.reference_field(dataset_cfg)

    agent = Agent(cfg)

    if cfg.role == "vqa":
        triples = list(data.vqa_examples(records, dataset_cfg["question_bank"]))
        specs = [(r, answer, q["question"]) for r, q, answer in triples]
    else:
        specs = [(r, r[reference], None) for r in records]

    def make_example(index: int):
        record, target, question = specs[index]
        return agent.example(record, fields, target=target, question=question)

    return cfg, dataset_cfg, agent, records, specs, make_example


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, dataset_cfg, agent, records, specs, make_example = build(args)
    output_dir = args.output_dir or f"runs/{cfg.name}"

    lora = agent.lora_config()
    print(f"agent          {cfg.name}  ({cfg.role})")
    print(f"backbone       {cfg.base_model}")
    print(f"prompt         {cfg.prompt}")
    print(f"lora           r={lora.r} alpha={lora.lora_alpha} targets={lora.target_modules}")
    print(f"loss           {cfg.train.get('loss')}")
    print(f"crop           {(cfg.extra.get('preprocess') or {}).get('crop', 'none')}")
    print(f"dataset        {dataset_cfg['name']}  ({len(records)} records -> {len(specs)} examples)")
    print(f"output         {output_dir}")

    if args.dry_run:
        if specs:
            example = make_example(0)
            batch = agent.collator()([example])
            labels = batch["labels"]
            kept = int((labels != -100).sum())
            print(f"first batch    input_ids {tuple(batch['input_ids'].shape)}, "
                  f"{kept}/{labels.numel()} label positions contribute to the loss")
        print("dry run: nothing was loaded or trained")
        return 0


    agent.load(adapter=False)
    trainer = SFTTrainer(
        model=agent.model,
        args=agent.sft_config(output_dir),
        train_dataset=Dataset.from_list([{"i": i} for i in range(len(specs))]),
        peft_config=lora,
        data_collator=lambda rows: agent.collator()([make_example(r["i"]) for r in rows]),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"adapter written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
