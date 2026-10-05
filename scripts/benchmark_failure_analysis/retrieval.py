"""Find explicit missing-source/identifier evidence, rather than treating all SQL errors as retrieval failures."""

import re

from .common import (
    answer_queries,
    arguments,
    finding,
    query_evidence,
    read_evidence,
    save_findings,
    sql_calls,
    tables,
)


def main():
    output = arguments(__doc__).output
    findings = []
    for attempt in read_evidence(output):
        if attempt["passed"] or attempt["status"] != "completed":
            continue
        queries = answer_queries(attempt)
        detail = {
            "queries": [query_evidence(t) for t in queries],
            "final_answer": attempt.get("output"),
        }
        relation = tables(attempt)
        if attempt["task"] == "task_1":
            text = (attempt.get("output") or "").lower()
            missing = [
                name
                for name in (
                    "power_quality_measurements_electrix",
                    "power_quality_measurements_procem",
                )
                if name not in text
            ]
            if missing:
                findings.append(
                    finding(
                        attempt,
                        "R1",
                        "retrieval",
                        "A required EV data source is absent from the answer.",
                        {**detail, "missing_sources": missing},
                    )
                )
        if attempt["task"] == "task_11":
            bad_queries = [
                t
                for t in sql_calls(attempt)
                if re.search(
                    r"device_id\s*=\s*'156'", t["arguments"]["sql"], re.IGNORECASE
                )
            ]
            if bad_queries and (not relation or relation[1].empty):
                findings.append(
                    finding(
                        attempt,
                        "R2",
                        "retrieval",
                        "Human label 156 is used instead of the database identifier device_156, yielding no data.",
                        {
                            **detail,
                            "bad_identifier_queries": [
                                query_evidence(t) for t in bad_queries
                            ],
                            "published_rows": len(relation[1]) if relation else None,
                        },
                    )
                )
        if attempt["task"] == "task_13" and relation:
            target, candidate, reference, path = relation
            if len(candidate) == 15 and len(target) == 315:
                findings.append(
                    finding(
                        attempt,
                        "R3",
                        "retrieval",
                        "Only the Electrix stations are returned; the 300 ProCem station rows are missing.",
                        {
                            **detail,
                            "reference": str(reference),
                            "candidate": str(path),
                            "expected_rows": len(target),
                            "actual_rows": len(candidate),
                            "candidate_devices": sorted(
                                set(candidate.iloc[:, 0].astype(str))
                            ),
                        },
                    )
                )
        if attempt["task"] == "task_14" and relation:
            target, candidate, reference, path = relation
            if len(candidate) < len(target) and any(
                "join" in t["arguments"]["sql"].lower() for t in queries
            ):
                findings.append(
                    finding(
                        attempt,
                        "R4",
                        "retrieval",
                        "A metadata inner join removes an observed meter without a matching catalog record.",
                        {
                            **detail,
                            "reference": str(reference),
                            "candidate": str(path),
                            "missing_device_ids": sorted(
                                set(target.iloc[:, 0]) - set(candidate.iloc[:, 0])
                            ),
                            "expected_rows": len(target),
                            "actual_rows": len(candidate),
                        },
                    )
                )
    save_findings(output, "retrieval", findings)
    print("retrieval findings", len(findings))


if __name__ == "__main__":
    main()
