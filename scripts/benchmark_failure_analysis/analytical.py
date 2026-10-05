"""Inspect numeric tables, timestamps, lag/sample choices, and aggregation signatures.

Output mismatch is identified separately from a confirmed implementation mechanism.
These checks diagnose saved answers; they do not edit or relax any evaluator.
"""

import json
import re
from collections import Counter
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.validation.evaluators.table import _column_positions, classify_table_match

from .common import (
    PENDING_TASKS,
    answer_queries,
    arguments,
    finding,
    query_evidence,
    read_evidence,
    save_findings,
    tables,
)


def table_detail(attempt, relation):
    target, actual, reference, path = relation
    positions = _column_positions(target, actual)
    result = {
        "reference": str(reference),
        "candidate": str(path),
        "expected_shape": list(target.shape),
        "actual_shape": list(actual.shape),
        "expected_columns": list(target.columns),
        "actual_columns": list(actual.columns),
        "column_positions": positions,
        "expected_first_rows": target.head(3).to_dict("records"),
        "actual_first_rows": actual.head(3).to_dict("records"),
    }
    if len(target) == len(actual) and positions is not None:
        mismatches = []
        for index, column in enumerate(target.columns):
            candidate = actual.iloc[:, positions[index]]
            if pd.api.types.is_numeric_dtype(
                target[column]
            ) and pd.api.types.is_numeric_dtype(candidate):
                delta = (
                    candidate.reset_index(drop=True)
                    - target[column].reset_index(drop=True)
                ).abs()
                mismatches.append(
                    {
                        "expected_column": column,
                        "actual_column": candidate.name,
                        "max_absolute_difference": delta.max(),
                    }
                )
        result["numeric_column_differences"] = mismatches
    return result


def truncated_timestamps(relation):
    target, actual, _, _ = relation
    positions = _column_positions(target, actual)
    if positions is None or len(actual) != len(target) or actual.empty:
        return []
    results = []
    for index, column in enumerate(target.columns):
        if not isinstance(target[column].iloc[0], str) or not re.match(
            r"\d{4}-\d{2}-\d{2}[ T]", target[column].iloc[0]
        ):
            continue
        try:
            expected = pd.to_datetime(target[column], utc=True, format="mixed")
            candidate = pd.to_datetime(
                actual.iloc[:, positions[index]], utc=True, format="mixed"
            )
        except (ValueError, TypeError):
            continue
        if (
            expected.reset_index(drop=True).dt.floor("s")
            == candidate.reset_index(drop=True)
        ).all() and not (
            expected.reset_index(drop=True) == candidate.reset_index(drop=True)
        ).all():
            results.append(
                {
                    "expected_column": column,
                    "candidate_column": actual.columns[positions[index]],
                    "max_lost_milliseconds": (
                        (
                            expected.reset_index(drop=True)
                            - candidate.reset_index(drop=True)
                        ).dt.total_seconds()
                        * 1000
                    ).max(),
                }
            )
    return results


def main():
    output = arguments(__doc__).output
    findings, table_checks = [], []
    reviewed = {}
    for mode in ("operational", "retrieval", "interpretation", "visualization"):
        document = json.loads((output / f"{mode}.json").read_text())
        for f in document["findings"]:
            reviewed.setdefault(f["span_id"], set()).add(f["mode"])
    for attempt in read_evidence(output):
        if attempt["passed"] or attempt["status"] != "completed":
            continue
        queries = answer_queries(attempt)
        sql = "\n".join(t["arguments"]["sql"] for t in queries).lower()
        compact = re.sub(r"\s+", "", sql.replace("`", ""))
        detail = {
            "task_question": attempt["input"],
            "expected_output": attempt["expected_output"],
            "final_answer": attempt.get("output"),
            "queries": [query_evidence(t) for t in queries],
        }
        relation = tables(attempt)
        matches = []
        task = attempt["task"]
        if relation:
            target, candidate, _, _ = relation
            detail["table"] = table_detail(attempt, relation)
            # Diagnostic only: policy continues to require at least two significant digits.
            coarse, _ = classify_table_match(
                target, candidate, minimum_significant_digits=1
            )
            accepted, _ = classify_table_match(
                target, candidate, minimum_significant_digits=2
            )
            detail["one_significant_digit_diagnostic"] = coarse
            detail["two_significant_digit_diagnostic"] = accepted
            table_checks.append(
                {
                    "span_id": attempt["span_id"],
                    "task": task,
                    **detail["table"],
                    "one_significant_digit_diagnostic": coarse,
                    "two_significant_digit_diagnostic": accepted,
                }
            )
            if coarse in {"exact_match", "match_up_to_ordering"} and accepted not in {
                "exact_match",
                "match_up_to_ordering",
            }:
                matches.append(
                    (
                        "A1",
                        "Saved table agrees only under rounding that retains fewer than two significant digits.",
                        "confirmed whole-table comparison",
                    )
                )
            truncation = truncated_timestamps(relation)
            if truncation:
                detail["truncated_timestamps"] = truncation
                matches.append(
                    (
                        "A2",
                        "Fractional seconds are discarded from the returned measurement/event boundaries.",
                        "confirmed timestamp comparison",
                    )
                )
            if task == "task_37" and not candidate.empty:
                start = pd.to_datetime(candidate.iloc[:, 0], utc=True, format="mixed")
                end = pd.to_datetime(candidate.iloc[:, 1], utc=True, format="mixed")
                if (start < pd.Timestamp("2025-03-02", tz="UTC")).any() or (
                    end >= pd.Timestamp("2025-03-03", tz="UTC")
                ).any():
                    matches.append(
                        (
                            "A3",
                            "Event extraction extends outside the requested UTC day, changing durations and means.",
                            "confirmed returned boundaries",
                        )
                    )
            if (
                task == "task_52"
                and "unix_timestamp" in sql
                and "unix_micros" not in sql
                and "microsecond" not in sql
            ):
                matches.append(
                    (
                        "A4",
                        "Integer-second duration calculation discards fractional duration before ranking.",
                        "reviewed published SQL signature",
                    )
                )
            if (
                task == "task_55"
                and "measure(voltage_thd_3_39" in compact
                and "avg(" not in compact
            ):
                matches.append(
                    (
                        "A5",
                        "The nonlinear metric is evaluated over the entire day rather than averaging separately computed ten-minute magnitudes.",
                        "reviewed published SQL signature",
                    )
                )
            if task == "task_23":
                matches.append(
                    (
                        "A6",
                        "The qualifying event relation differs: filtering/continuity/grouping or temporal ordering changes the requested peaks.",
                        "confirmed relation discrepancy; inspect producer SQL for subtype",
                    )
                )
            if task == "task_48":
                matches.append(
                    (
                        "A7",
                        "Harmonic aggregation, conversion, baseline weighting or day assignment produces different event rows/factors.",
                        "confirmed relation discrepancy; exact formula subtype requires SQL review",
                    )
                )
            if task == "task_51" and "interpretation" not in reviewed.get(
                attempt["span_id"], set()
            ):
                rounded = candidate.copy()
                for column in rounded.select_dtypes(include="number").columns:
                    rounded[column] = rounded[column].round(1)
                round_match, _ = classify_table_match(target, rounded)
                if round_match in {"exact_match", "match_up_to_ordering"}:
                    matches.append(
                        (
                            "A18",
                            "Correct current values are returned without the explicitly requested one-decimal rounding.",
                            "confirmed after rounding numeric cells only",
                        )
                    )
                else:
                    matches.append(
                        (
                            "A8",
                            "The saved heating-current relation differs in metric values, selected rows or source identifiers.",
                            "confirmed relation discrepancy; exact subtype retained in evidence",
                        )
                    )
            if task == "task_38":
                if candidate.iloc[:, 0].astype(str).str.fullmatch(r"\d{2}:\d{2}").all():
                    matches.append(
                        (
                            "A19",
                            "Returned hour labels omit the calendar date required for UTC timestamps.",
                            "confirmed published timestamp fields",
                        )
                    )
                elif "price" in compact:
                    matches.append(
                        (
                            "A20",
                            "Hourly charging power is paired with the next price hour instead of the containing price hour.",
                            "reviewed SQL and returned price sequence",
                        )
                    )
            if (
                task == "task_39"
                and "ev_samples=900" in compact
                and "solar_samples=900" in compact
            ):
                matches.append(
                    (
                        "A21",
                        "An unrequested exact-900-readings condition discards valid quarter-hour intervals.",
                        "confirmed published filter",
                    )
                )
            if task == "task_52" and re.search(r"count\(\*\)-1asduration", compact):
                matches.append(
                    (
                        "A22",
                        "Elapsed seconds are approximated by measurement count minus one.",
                        "confirmed published duration expression",
                    )
                )
            if task == "task_52" and "interval 1 second" in sql:
                matches.append(
                    (
                        "A23",
                        "An exact one-second adjacency test splits consecutive jittered measurements into shorter events.",
                        "reviewed published continuity predicate",
                    )
                )
            if (
                task == "task_53"
                and "pow(total_distortion_pct,2)-harmonic_sq" in compact
            ):
                matches.append(
                    (
                        "A24",
                        "Discrepancy is computed as the square root of a difference of squares instead of the absolute difference of magnitudes.",
                        "confirmed published algebra",
                    )
                )
        if task == "task_33":
            zone = ZoneInfo("Europe/Helsinki")
            begin = datetime(2025, 3, 30, tzinfo=zone).astimezone(UTC)
            end = datetime(2025, 3, 31, tzinfo=zone).astimezone(UTC)
            detail["expected_local_day_utc"] = [begin.isoformat(), end.isoformat()]
            matches.append(
                (
                    "A9",
                    "The returned energy cost differs; UTC window, price-hour matching and power-to-energy conversion are checked in the retained SQL.",
                    "confirmed numeric discrepancy; causal signatures below",
                )
            )
            if (
                "2025-03-29 23:00:00" in sql
                or "2025-03-30 22:00:00" in sql
                or "2025-03-31 00:00:00" in sql
            ):
                matches.append(
                    (
                        "A10",
                        "The SQL uses a UTC boundary outside the 23-hour Helsinki local day.",
                        "reviewed final-query signature",
                    )
                )
            if re.search(r"timestamp\s*-\s*interval\s+1\s+second", sql):
                matches.append(
                    (
                        "A11",
                        "Power readings are shifted to a different hourly price interval.",
                        "reviewed final-query signature",
                    )
                )
        if task == "task_35":
            if "-0.064868" in (attempt.get("output") or "") and "0.048392" in (
                attempt.get("output") or ""
            ):
                matches.append(
                    (
                        "A12",
                        "Same-minute correlation reuses the 235 lagged pairs instead of its separate 240-pair sample.",
                        "confirmed output/sample-count signature",
                    )
                )
            if re.search(
                r"wind[^\n]{0,45}=\s*(?:power|p\.)[^\n]{0,45}\+\s*(?:interval\s+)?5",
                sql,
            ):
                matches.append(
                    (
                        "A13",
                        "Five-minute pairing reverses the requested ventilation-later lag direction.",
                        "reviewed SQL signature",
                    )
                )
            if "2025-03-17 12:05:00" in sql:
                matches.append(
                    (
                        "A25",
                        "The ventilation series extends five minutes past the explicitly exclusive 12:00 endpoint.",
                        "confirmed final-query boundary",
                    )
                )
        if (
            task == "task_34"
            and "22.18" in (attempt.get("output") or "")
            and "avg(pw) as demand" in sql
        ):
            matches.append(
                (
                    "A26",
                    "Meter readings are pooled into an average instead of summing the two demand components.",
                    "confirmed final-query aggregation",
                )
            )
        if task == "task_36" and "pw-rolling_avg_w>=2000" in compact:
            matches.append(
                (
                    "A14",
                    "An unrequested 2,000 W increase threshold removes the correct event.",
                    "confirmed published threshold",
                )
            )
        if task == "task_47":
            matches.append(
                (
                    "A15",
                    "Differencing, lag direction/context or clock alignment yields wrong delay/correlation rows.",
                    "confirmed relation discrepancy; producer SQL retained",
                )
            )
        if task == "task_54":
            matches.append(
                (
                    "A16",
                    "Correlation uses a different matched sample/grid or lag than the required one-second means.",
                    "confirmed relation discrepancy; producer SQL retained",
                )
            )
        if task == "task_17" and "21:44" in (attempt.get("output") or ""):
            matches.append(
                (
                    "A17",
                    "A minute-grid endpoint replaces the measurement-level rolling-window maximum at 21:43.",
                    "confirmed selected-time discrepancy; query retained",
                )
            )
        if task in {"task_42", "task_44"}:
            chart_data = json.loads((output / "visualization.json").read_text())
            chart = next(
                c for c in chart_data["charts"] if c["span_id"] == attempt["span_id"]
            )
            if task == "task_42" and chart["checks"].get("tenant_series_halved"):
                matches.append(
                    (
                        "A27",
                        "The tenant active-power series is halved by applying energy-counter correction metadata to instantaneous power.",
                        "confirmed complete plotted-series comparison",
                    )
                )
            if (
                task == "task_44"
                and chart["checks"].get("hours_complete")
                and not chart["checks"].get(
                    "hourly_percentages_match_allowing_rounding"
                )
            ):
                matches.append(
                    (
                        "A28",
                        "The plotted hourly percentages differ beyond consistent display rounding; activity thresholds or denominators change.",
                        "confirmed chart-data comparison; exact predicate retained in SQL",
                    )
                )
            detail["chart_data_checks"] = chart["checks"]
        for rule, explanation, confidence in matches:
            findings.append(
                finding(attempt, rule, "analytical", explanation, detail, confidence)
            )
        if (
            not matches
            and task not in PENDING_TASKS
            and not any("image_path" in e for e in attempt["expected_output"])
            and not reviewed.get(attempt["span_id"])
        ):
            findings.append(
                finding(
                    attempt,
                    "A0",
                    "analytical",
                    "The saved numerical relation/answer differs from the corrected reference; the specific mechanism remains unresolved.",
                    detail,
                    "confirmed evaluator discrepancy; root cause unresolved",
                )
            )
    save_findings(
        output,
        "analytical",
        findings,
        {
            "table_checks": table_checks,
            "note": "A0 and broad relation signatures are observations of wrong data, not proof of a unique calculation mistake. No scores changed.",
        },
    )
    print("analytical findings", dict(Counter(f["rule"] for f in findings)))


if __name__ == "__main__":
    main()
