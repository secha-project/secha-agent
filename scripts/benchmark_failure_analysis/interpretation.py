"""Detect evidenced changes to the requested source, metric, sign, or output grain."""

import re

from .common import (
    answer_queries,
    arguments,
    finding,
    query_evidence,
    read_evidence,
    save_findings,
    tables,
)


def main():
    output = arguments(__doc__).output
    findings = []
    for attempt in read_evidence(output):
        if attempt["passed"] or attempt["status"] != "completed":
            continue
        queries = answer_queries(attempt)
        sql = "\n".join(t["arguments"]["sql"] for t in queries).lower()
        compact = re.sub(r"\s+", "", sql.replace("`", ""))
        text = attempt.get("output") or ""
        detail = {
            "task_question": attempt["input"],
            "final_answer": text,
            "queries": [query_evidence(t) for t in queries],
        }
        matches = []
        task = attempt["task"]
        if task == "task_18" and "kampusareenawatercooling02" in text.lower():
            matches.append(
                (
                    "I1",
                    "Search scope includes ProCem rather than the intended Electrix dataset; original question had a typo.",
                )
            )
        if task == "task_32" and re.search(r"corr\([^)]*pl[123]w", compact):
            matches.append(
                (
                    "I2",
                    "Per-phase active power is substituted for total three-phase power; pending wording clarification.",
                )
            )
        if task == "task_34" and re.fullmatch(r"\s*9\.98\s*", text) and "solar" in sql:
            matches.append(
                (
                    "I3",
                    "Solar generation is added to the demand denominator, changing the requested metered-demand boundary.",
                )
            )
        if (
            task == "task_46"
            and "tdul" in sql
            and not any(term in sql for term in ("u1l", "voltage_thd", "u3l"))
        ):
            matches.append(
                (
                    "I4",
                    "ABS of the relative distortion percentage is used instead of absolute harmonic magnitude in volts.",
                )
            )
        if (
            task == "task_51"
            and ("tdil" in sql or "true_rms" in sql)
            and "equivalent_winding_heating" not in sql
        ):
            matches.append(
                (
                    "I5",
                    "True RMS or an unweighted total-distortion adjustment replaces the equivalent winding-heating current.",
                )
            )
        if (
            task == "task_53"
            and re.search(r"\bul[123]v\b", sql)
            and not re.search(r"\bdl[123]\b", sql)
        ):
            matches.append(
                (
                    "I6",
                    "Measured total RMS voltage replaces the meter's reported distortion in the discrepancy definition.",
                )
            )
        if task == "task_49" and re.search(r"u1l[123]v\*i1l[123]a", compact):
            matches.append(
                (
                    "I7",
                    "A fundamental U×I convention replaces the reference's fundamental P/Q apparent-power reconstruction.",
                )
            )
        if task == "task_39" and (
            "-solar" in compact
            or re.search(r"(?:solar_mean_w|pv|solar_pw|solar_w)<0", compact)
        ):
            matches.append(
                (
                    "I8",
                    "The solar generation sign convention is reversed, changing interval selection.",
                )
            )
        relation = tables(attempt)
        if (
            task == "task_23"
            and relation
            and len(relation[1]) <= 4
            and re.search(r"groupbydevice_id(?:orderby|$)", compact)
        ):
            matches.append(
                (
                    "I9",
                    "Distinct flicker events are collapsed into one month-spanning range per device.",
                )
            )
        if task == "task_34" and "35.99" in text:
            matches.append(
                (
                    "I10",
                    "Only the maintenance demand is used; tenant demand is omitted from the building boundary.",
                )
            )
        for rule, explanation in matches:
            findings.append(
                finding(
                    attempt,
                    rule,
                    "interpretation",
                    explanation,
                    detail,
                    "reviewed SQL/output signature; task-specific predicate",
                )
            )
    save_findings(
        output,
        "interpretation",
        findings,
        {
            "note": "Interpretation labels describe the selected question/metric, not an inference about hidden intent. Pending tasks are excluded from final-cause totals."
        },
    )
    print("interpretation findings", len(findings))


if __name__ == "__main__":
    main()
