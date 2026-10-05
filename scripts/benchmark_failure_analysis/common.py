import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark-results/power-quality-monitoring"
DEFAULT_OUTPUT = ROOT / "output/benchmark-failure-analysis-corrected-2026-10-05"
REFERENCE = ROOT / "src/validation/benchmark/power-quality-monitoring"
PENDING_TASKS = {"task_18", "task_31", "task_32"}


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def read_evidence(output):
    with (output / "evidence.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def artifact(attempt, key):
    value = attempt.get("attributes", {}).get(key)
    if not value:
        return None
    return SOURCE / attempt["run"] / "artifacts" / Path(value).name


def sql_calls(attempt, successful_only=True):
    return [
        tool
        for tool in attempt["tools"]
        if tool["name"] == "execute_tool query_data"
        and (not successful_only or tool["status"] != "ERROR")
        and isinstance(tool["arguments"], dict)
        and isinstance(tool["arguments"].get("sql"), str)
    ]


def answer_queries(attempt):
    """Bind published artifacts to their producer; scalar answers keep last two queries.

    Do not attribute an abandoned exploratory query to a published table/chart.
    """
    producers = {
        t["result"]["output_ref"]["output_id"]: t
        for t in sql_calls(attempt)
        if isinstance(t["result"], dict)
        and isinstance(t["result"].get("output_ref"), dict)
    }
    published = []
    for tool in attempt["tools"]:
        if tool["status"] == "ERROR" or tool["name"] not in {
            "execute_tool display_table_output",
            "execute_tool visualize_table_output",
        }:
            continue
        args = tool["arguments"]
        if isinstance(args, dict):
            ref = args.get("ref", args)
            producer = (
                producers.get(ref.get("output_id")) if isinstance(ref, dict) else None
            )
            if producer:
                published.append(producer)
    return published[-1:] if published else sql_calls(attempt)[-2:]


def query_evidence(tool):
    return {
        "tool_span_id": tool["span_id"],
        "sql": tool["arguments"]["sql"],
        "result": tool["result"],
    }


def tables(attempt):
    import pandas as pd

    expected = next((e for e in attempt["expected_output"] if "table_path" in e), None)
    path = artifact(attempt, "csv_path")
    if not expected or not path or not path.is_file():
        return None
    reference = REFERENCE / expected["table_path"]
    return pd.read_csv(reference), pd.read_csv(path), reference, path


def failure(attempt):
    return not attempt["passed"]


def identity(attempt):
    return {
        key: attempt[key]
        for key in (
            "run",
            "config",
            "task",
            "name",
            "trace_id",
            "span_id",
            "status",
            "passed",
        )
    }


def finding(attempt, rule, mode, explanation, evidence, confidence="confirmed"):
    return {
        **identity(attempt),
        "rule": rule,
        "mode": mode,
        "explanation": explanation,
        "evidence": evidence,
        "confidence": confidence,
        "pending_task": attempt["task"] in PENDING_TASKS,
        "affects_final": failure(attempt),
    }


def save_findings(output, mode, findings, extra=None):
    write_json(
        output / f"{mode}.json",
        {
            "mode": mode,
            "findings": findings,
            "unique_attempts": len({f["span_id"] for f in findings}),
            "rules": dict(Counter(f["rule"] for f in findings)),
            **(extra or {}),
        },
    )


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
