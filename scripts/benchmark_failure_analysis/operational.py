"""Extract terminal tool misuse, missing publication, and recovered tool errors."""

import json
import re
from collections import Counter

from .common import (
    arguments,
    artifact,
    decode,
    finding,
    query_evidence,
    read_evidence,
    save_findings,
    sql_calls,
)


def terminal_rule(attempt):
    stack = attempt.get("error_stacktrace", "")
    if "Input should be an object" in stack and "input_type=str" in stack:
        return (
            "O1",
            "operational",
            "A nested table ref is serialized as a string instead of an object.",
        )
    if "INTERNAL_ERROR" in stack:
        return "P1", "platform", "Spark internal metric-view planning failure."
    if "SparkActionTimeoutError" in stack:
        return "P2", "platform", "Spark action timeout."
    if "UnknownTableOutputError" in stack:
        return "O2", "operational", "Unknown or corrupted table-output identifier."
    if "Tool 'query_data' exceeded" in stack:
        return "O3", "operational", "Invalid SQL exhausts the query correction budget."
    if "Tool 'visualize_table_output' exceeded" in stack:
        return (
            "O4",
            "operational",
            "Invalid visualization tool input exhausts the correction budget.",
        )
    return (
        "U1",
        "unresolved",
        "Unrecognized terminal exception; inspect the saved stack.",
    )


def malformed_calls(attempt):
    calls = []
    for model in attempt["models"]:
        for message in model.get("messages") or []:
            if not isinstance(message, dict):
                continue
            for part in message.get("parts", []):
                args = decode(part.get("arguments"))
                if (
                    part.get("type") == "tool_call"
                    and isinstance(args, dict)
                    and isinstance(args.get("ref"), str)
                ):
                    calls.append(
                        {
                            "model_span_id": model["span_id"],
                            "tool": part.get("name"),
                            "arguments": args,
                        }
                    )
    return calls


def main():
    output = arguments(__doc__).output
    findings, terminals, events = [], [], []
    for attempt in read_evidence(output):
        if attempt["status"] == "failed":
            rule, mode, explanation = terminal_rule(attempt)
            error_tools = [t for t in attempt["tools"] if t["status"] == "ERROR"]
            detail = {
                "error_message": attempt.get("error_message"),
                "error_stacktrace": attempt.get("error_stacktrace"),
                "last_error_tools": error_tools[-3:],
            }
            if rule == "O1":
                detail["malformed_model_calls"] = malformed_calls(attempt)
                # Schema-invalid calls never enter execute_tool spans.
                detail["model_call_evidence_available"] = bool(
                    detail["malformed_model_calls"]
                )
            row = finding(attempt, rule, mode, explanation, detail)
            terminals.append(row)
            findings.append(row)
        table_expected = any(
            "table_path" in expected for expected in attempt["expected_output"]
        )
        path = artifact(attempt, "csv_path")
        if (
            table_expected
            and not attempt["passed"]
            and attempt["status"] == "completed"
            and not (path and path.is_file())
        ):
            publication = [
                t
                for t in attempt["tools"]
                if t["name"] == "execute_tool display_table_output"
                and t["status"] != "ERROR"
            ]
            queries = sql_calls(attempt)
            if not publication:
                findings.append(
                    finding(
                        attempt,
                        "O5",
                        "operational",
                        "Requested table is not published through display_table_output.",
                        {
                            "csv_path": str(path) if path else None,
                            "successful_publication_calls": 0,
                            "successful_queries": [
                                query_evidence(t) for t in queries[-2:]
                            ],
                            "final_answer": attempt.get("output"),
                        },
                    )
                )
        for tool in attempt["tools"]:
            if tool["status"] != "ERROR":
                continue
            text = json.dumps([tool["message"], tool["result"], tool["events"]])
            codes = sorted(set(re.findall(r"\[([A-Z][A-Z_]*(?:\.[A-Z_]*)?)\]", text)))
            events.append(
                {
                    "span_id": attempt["span_id"],
                    "run": attempt["run"],
                    "task": attempt["task"],
                    "passed": attempt["passed"],
                    "tool": tool["name"],
                    "tool_span_id": tool["span_id"],
                    "codes": codes,
                    "arguments": tool["arguments"],
                    "result": tool["result"],
                    "message": tool["message"],
                }
            )
    save_findings(
        output,
        "operational",
        findings,
        {
            "terminal_counts": dict(Counter(f["rule"] for f in terminals)),
            "tool_error_events": events,
            "tool_error_codes": dict(
                Counter(
                    code for event in events for code in event["codes"]
                ).most_common()
            ),
            "event_count_note": "Recovered tool errors are events, not final-answer failure classifications.",
        },
    )
    print("terminal causes", dict(Counter(f["rule"] for f in terminals)))


if __name__ == "__main__":
    main()
