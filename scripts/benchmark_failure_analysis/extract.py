"""Join only corrected report attempts to their SQLite descendants, read-only."""

import json
import sqlite3
from collections import Counter, defaultdict

from .common import SOURCE, arguments, decode, sha256, write_json


def main():
    output = arguments(__doc__).output
    output.mkdir(parents=True, exist_ok=True)
    attempts, reports = [], []
    paths = sorted(SOURCE.glob("*/report-evaluator-fixed.json"))
    if len(paths) != 16:
        raise ValueError(
            f"Expected the 16 completed corrected reports; found {len(paths)}"
        )
    for path in paths:
        report = json.loads(path.read_text())
        meta = report["experiment_metadata"]
        original = json.loads((path.parent / "report.json").read_text())
        original_attempts = {
            a["span_id"]: a
            for category in original["results"]["categories"].values()
            for task in category["tasks"].values()
            for a in task["attempts"]
        }
        reports.append(
            {
                "run": path.parent.name,
                "path": str(path),
                "sha256": sha256(path),
                "original_sha256": sha256(path.parent / "report.json"),
                "metadata": meta,
                "overall": report["results"]["overall"],
            }
        )
        for category, group in report["results"]["categories"].items():
            for task in group["tasks"].values():
                for attempt in task["attempts"]:
                    attempts.append(
                        {
                            "run": path.parent.name,
                            "config": meta["run_config"],
                            "category": category,
                            "task": task["name"],
                            "input": task["input"],
                            "expected_output": task["expected_output"],
                            "original_passed": original_attempts[attempt["span_id"]][
                                "passed"
                            ],
                            **attempt,
                        }
                    )
    assert len(attempts) == 2640 and len({a["span_id"] for a in attempts}) == 2640
    database = SOURCE / "phoenix.db"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    spans = {}
    for row in connection.execute(
        "SELECT span_id,parent_id,name,span_kind,start_time,end_time,status_code,status_message FROM spans"
    ):
        spans[row[0]] = dict(
            zip(
                (
                    "span_id",
                    "parent_id",
                    "name",
                    "kind",
                    "start",
                    "end",
                    "status",
                    "message",
                ),
                row,
            )
        )
    case_ids = {
        span_id
        for span_id, span in spans.items()
        if span["name"] == "case: {case_name}"
    }
    canonical = {a["span_id"] for a in attempts}
    parents = {}

    def ancestor(span_id):
        path, seen = [], set()
        while span_id and span_id not in seen:
            if span_id in parents:
                found = parents[span_id]
                break
            seen.add(span_id)
            path.append(span_id)
            if span_id in case_ids:
                found = span_id
                break
            span_id = spans.get(span_id, {}).get("parent_id")
        else:
            found = None
        for member in path:
            parents[member] = found
        return found

    for span_id in spans:
        ancestor(span_id)
    tools, models = defaultdict(list), defaultdict(list)
    for span_id, raw, events in connection.execute(
        "SELECT span_id,json(attributes),json(events) FROM spans WHERE name LIKE 'execute_tool %'"
    ):
        owner = parents[span_id]
        if owner not in canonical:
            continue
        attributes = json.loads(raw)
        call = attributes.get("gen_ai", {}).get("tool", {}).get("call", {})
        tools[owner].append(
            {
                **spans[span_id],
                "arguments": decode(call.get("arguments")),
                "result": decode(call.get("result")),
                "events": json.loads(events),
            }
        )
    for span_id, raw in connection.execute(
        "SELECT span_id,json_extract(attributes,'$.gen_ai.output.messages') FROM spans WHERE name LIKE 'chat %'"
    ):
        owner = parents[span_id]
        if owner in canonical:
            models[owner].append({**spans[span_id], "messages": decode(raw)})
    connection.close()
    with (output / "evidence.jsonl").open("w") as handle:
        for attempt in attempts:
            attempt["trace_present"] = attempt["span_id"] in case_ids
            attempt["tools"] = sorted(
                tools[attempt["span_id"]], key=lambda t: (t["start"], t["span_id"])
            )
            attempt["models"] = sorted(
                models[attempt["span_id"]], key=lambda t: (t["start"], t["span_id"])
            )
            handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
    write_json(output / "reports.json", reports)
    write_json(
        output / "inventory.json",
        {
            "database": str(database),
            "database_size": database.stat().st_size,
            "database_sha256": sha256(database),
            "spans": len(spans),
            "case_spans": len(case_ids),
            "attempts": len(attempts),
            "outcomes": dict(
                Counter("pass" if a["passed"] else a["status"] for a in attempts)
            ),
            "evaluator_corrections": sum(
                a["passed"] and not a["original_passed"] for a in attempts
            ),
            "missing_traces": [
                {"run": a["run"], "name": a["name"], "span_id": a["span_id"]}
                for a in attempts
                if not a["trace_present"]
            ],
            "supplemental_case_spans_excluded": len(case_ids - canonical),
            "tool_spans": sum(len(a["tools"]) for a in attempts),
            "tool_error_spans": sum(
                t["status"] == "ERROR" for a in attempts for t in a["tools"]
            ),
            "attempts_with_tool_errors": sum(
                any(t["status"] == "ERROR" for t in a["tools"]) for a in attempts
            ),
            "passing_attempts_with_tool_errors": sum(
                a["passed"] and any(t["status"] == "ERROR" for t in a["tools"])
                for a in attempts
            ),
            "scrubbed_argument_attempts": [
                a["span_id"]
                for a in attempts
                if any("Scrubbed" in json.dumps(t["arguments"]) for t in a["tools"])
            ],
        },
    )
    print(f"Extracted {len(attempts)} corrected attempts, read-only, to {output}")


if __name__ == "__main__":
    main()
