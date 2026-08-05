#!/usr/bin/env python3
"""Merge historical and DAgGER datasets with provenance and V8 validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_sources(values: list[str], option: str) -> list[tuple[str, Path]]:
    sources = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use stage=path: {value}")
        stage, path = value.split("=", 1)
        if not stage or not path:
            raise ValueError(f"invalid {option}: {value}")
        sources.append((stage, Path(path)))
    return sources


def merge(sources: list[tuple[str, Path]], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts = Counter()
    teacher_profiles = Counter()
    phases = Counter()
    teacher_action_types = Counter()
    records = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for stage, path in sources:
            if not path.exists():
                raise FileNotFoundError(path)
            for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
                record = json.loads(line)
                if stage != "historical":
                    if record.get("teacher_profile_id") != "v008":
                        raise ValueError(f"{path}:{line_number}: DAgGER teacher must be v008")
                    if "teacher_scores" not in record or "heuristic_scores" not in record:
                        raise ValueError(f"{path}:{line_number}: missing V8 teacher scores")
                    if record.get("chosen_action_index") != record.get("teacher_action_index"):
                        raise ValueError(f"{path}:{line_number}: chosen_action is not the V8 teacher label")
                    teacher_profiles[record["teacher_profile_id"]] += 1
                    teacher_action_types[record.get("teacher_action_type", "unknown")] += 1
                phases[record.get("observation", {}).get("phase", "unknown")] += 1
                record["dataset_source"] = stage
                record["dagger_stage"] = stage if stage != "historical" else None
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                counts[stage] += 1
                records += 1
    temporary.replace(output)
    return {
        "output": str(output),
        "records": records,
        "records_by_source": dict(counts),
        "teacher_profiles": dict(teacher_profiles),
        "phases": dict(phases),
        "teacher_action_types": dict(teacher_action_types),
        "sources": [{"stage": stage, "path": str(path)} for stage, path in sources],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="STAGE=PATH")
    parser.add_argument("--validation-source", action="append", required=True, metavar="STAGE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        sources = parse_sources(args.source, "--source")
        validation_sources = parse_sources(args.validation_source, "--validation-source")
        manifest = merge(sources, args.output)
        validation_manifest = merge(validation_sources, args.validation_output)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    combined = {"training": manifest, "validation": validation_manifest}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, sort_keys=True))


if __name__ == "__main__":
    main()
