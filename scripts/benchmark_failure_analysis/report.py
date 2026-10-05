"""Write a concise failure report and a complete classification ledger, without SQL dumps."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    PENDING_TASKS,
    ROOT,
    arguments,
    artifact,
    read_evidence,
    sha256,
    write_csv,
    write_json,
)


def main():
    output = arguments(__doc__).output
    attempts = read_evidence(output)
    inventory = json.loads((output / "inventory.json").read_text())
    findings = defaultdict(list)
    for mode in (
        "operational",
        "retrieval",
        "interpretation",
        "analytical",
        "visualization",
    ):
        for row in json.loads((output / f"{mode}.json").read_text())["findings"]:
            findings[row["span_id"]].append(row)
    ledger, summary_counts = [], Counter()
    for attempt in attempts:
        rows = findings[attempt["span_id"]]
        modes = {row["mode"] for row in rows}
        if attempt["passed"]:
            primary = "pass"
        elif attempt["task"] in PENDING_TASKS:
            primary = "pending task rerun"
        else:
            primary = next(
                (
                    mode
                    for mode in (
                        "platform",
                        "retrieval",
                        "operational",
                        "interpretation",
                        "analytical",
                        "visualization",
                    )
                    if mode in modes
                ),
                "unresolved",
            )
        summary_counts[primary] += 1
        ledger.append(
            {
                "run": attempt["run"],
                "task": attempt["task"],
                "attempt": attempt["name"],
                "passed": attempt["passed"],
                "status": attempt["status"],
                "summary_category": primary,
                "secondary_categories": "; ".join(sorted(modes - {primary})),
                "rules": "; ".join(sorted({row["rule"] for row in rows})),
                "confidence": "; ".join(sorted({row["confidence"] for row in rows})),
                "trace_id": attempt["trace_id"],
                "case_span_id": attempt["span_id"],
                "trace_present": attempt["trace_present"],
                "artifact": str(
                    artifact(attempt, "csv_path")
                    or artifact(attempt, "image_path")
                    or ""
                ),
            }
        )
    assert len(ledger) == 2640 and sum(summary_counts.values()) == 2640
    assert summary_counts["unresolved"] == 0
    reports = json.loads((output / "reports.json").read_text())
    assert all(sha256(Path(report["path"])) == report["sha256"] for report in reports)
    assert all(
        sha256(Path(report["path"]).parent / "report.json") == report["original_sha256"]
        for report in reports
    )
    write_csv(output / "attempt_classifications.csv", ledger)
    rule_counts = Counter(
        row["rule"]
        for group in findings.values()
        for row in group
        if row["task"] not in PENDING_TASKS
    )
    presence = {
        mode: sum(
            not a["passed"]
            and a["task"] not in PENDING_TASKS
            and any(row["mode"] == mode for row in findings[a["span_id"]])
            for a in attempts
        )
        for mode in (
            "operational",
            "interpretation",
            "retrieval",
            "analytical",
            "visualization",
        )
    }
    write_json(
        output / "summary.json",
        {
            "inventory": inventory,
            "summary_categories": dict(summary_counts),
            "nonexclusive_category_counts": presence,
            "rule_counts_excluding_pending_tasks": dict(rule_counts),
            "precedence": [
                "platform",
                "retrieval",
                "operational",
                "interpretation",
                "analytical",
                "visualization",
            ],
            "classification_note": "One summary label per failed attempt, with secondary findings retained. Broad analytical signatures confirm differing data but do not establish a unique internal cause.",
            "sources_unchanged": True,
        },
    )
    counts = summary_counts
    report = f"""FAILURE ANALYSIS — CORRECTED POWER-QUALITY BENCHMARK
5 October 2026

Scope
16 configurations, 55 tasks, three retained repetitions: 2,640 attempts.
Corrected outcomes: 1,871 passes, 536 completed assertion failures, and 233 execution failures.
The evaluator fixes recovered 166 attempts. Those recovered answers are excluded from the failure analysis.

Tasks 18, 31 and 32 are awaiting reruns after wording corrections. Their {counts["pending task rerun"]} currently failed attempts are set aside rather than counted as established agent errors. Of the other 702 failed attempts, eight are platform failures and 694 fall under the requested taxonomy.

Working classification of the remaining failures
Operational:    {counts["operational"]}
Interpretation: {counts["interpretation"]}
Retrieval:      {counts["retrieval"]}
Analytical:     {counts["analytical"]}
Visualization:  {counts["visualization"]}
Platform:      {counts["platform"]}

Each attempt receives one summary label. Some have secondary findings; the complete ledger preserves them. When mechanisms overlap, missing data takes precedence over missing publication, and calculation errors take precedence over the resulting bad plot. The labels are trace-based working classifications, not a claim that every agent's internal cause is uniquely identifiable.

1. OPERATIONAL ERROR — incorrect use of the provided tools

The dominant failure is malformed table publication: 217 attempts send a nested table reference as a string instead of the object required by the tool. All 217 are Qwen attempts. In a task 11 example, the correct 360-row dataset is retrieved, but publication fails twice even after the agent recognizes the required argument shape. This is a tool-interface failure after successful retrieval.

Other terminal failures comprise three corrupted/unknown output references, three SQL correction-budget exhaustions, and one visualization correction-budget exhaustion. One of the SQL failures is on pending task 31 and is excluded from the main counts above.

Fourteen completed table answers have no published CSV. These occur in tasks 11, 24, 38, 46, 51 and 52. Two task 11 cases also have a retrieval failure and are summarized under retrieval. Returning a Markdown table or claiming that a table is available does not deliver the required artifact.

2. INTERPRETATION ERROR — answering a different question

The agent selects a different metric or scope despite having relevant data. Twelve task 34 answers use gross demand including solar generation in the denominator and return 9.98 instead of the intended 11.09. Some even retrieve both results and select the gross-demand result. Another answer uses maintenance demand alone.

In task 46, 22 failed published answers use the relative distortion percentage channels rather than harmonic magnitude in volts. Task 51 includes total-distortion or true-RMS substitutions for equivalent winding-heating current. Task 53 includes total-RMS-voltage comparisons where the reference asks for distortion discrepancy. Other examples reverse the solar sign convention or collapse separate flicker events into a month-spanning range.

Tasks 18 and 32 show additional source/power-definition substitutions, but their original wording was ambiguous. They remain outside the established-error counts pending reruns.

3. RETRIEVAL ERROR — failing to obtain needed platform data

Twelve completed failures have clear missing-data evidence:
- Task 1: one answer omits a required EV data source.
- Task 11: three answers use identifier 156 instead of device_156 and conclude that data are absent or publish an empty relation.
- Task 13: six answers retrieve only 15 Electrix rows and omit the 300 ProCem EV-station rows; the full answer has 315 rows.
- Task 14: two metadata inner joins drop an observed meter, returning three devices instead of four.

An empty relation is not evidence that the requested measurements do not exist. Identifier resolution, source completeness and missing metadata need separate checks.

4. ANALYTICAL ERROR — incorrect handling of data

These errors concern time windows, sample selection, aggregation, event grouping and precision:
- Task 35: seven answers reuse the restricted lagged sample for the same-minute correlation; another extends the ventilation series past 12:00, and one reverses the lag direction.
- Task 33: local-day/DST boundaries, price-hour alignment and power-to-energy conversion produce the wrong cost. The correct Helsinki local day is a 23-hour UTC interval.
- Task 34: one answer pools the two demand meters into an average instead of summing their demand.
- Task 36: an unrequested 2,000 W threshold removes the correct event.
- Tasks 23 and 52: filtering or exact timestamp-adjacency rules change event continuity; elapsed duration is sometimes approximated by sample count or truncated before ranking.
- Task 39: exact-900-readings filters discard otherwise valid quarter-hour intervals.
- Tasks 47 and 54: changing differencing, lag context or clock alignment changes the matched sample and correlations.
- Tasks 48 and 55: nonlinear harmonic aggregation is performed at the wrong stage. Seven task 55 failures evaluate the harmonic measure across the whole day instead of averaging separately computed ten-minute magnitudes.
- Twenty-one tables remain consistent only with rounding below the approved two-significant-digit floor. These remain rejected under the agreed policy. Four task 51 answers return correct current values without the explicitly requested one-decimal rounding.

For broad task-family findings, the saved relation confirms incorrect output, while the exact formula subtype remains a review judgment. The scripts retain this distinction in the confidence field.

5. VISUALIZATION ERROR — incorrect representation of data

There are 480 requested-chart attempts: 242 passes, 236 completed rejections, and two execution failures. Three rejected charts also have confirmed incorrect plotted percentages and are summarized as analytical errors; this leaves 233 with visualization as their summary category. All original visualization verdicts are retained.

The clearest representation failures are:
- Task 26: all 48 plots have the correct twelve histogram bin counts. Forty-two use categorical bandwidth for bars on a continuous axis, producing hairlines rather than visible bins.
- Task 29: all 47 rendered plots have the correct 57 phase/harmonic means, allowing display rounding. They stack or overlay the phases instead of showing the requested adjacent bars.
- Task 27: 41 rejections involve weekday ordering, tile visibility or labels.
- Tasks 43 and 45: clipped or missing legend entries prevent users from identifying series.
- Tasks 42 and 44: positioning/stacking, category layout or activity percentages can make the chart incorrect despite successful data retrieval.

Forty-four charts fail only aesthetics. These are presentation failures under the retained verdicts, not evidence of incorrect calculation. Image-judge labels are observations: a data_correct rejection alone does not prove that the underlying SQL values were wrong.

Unexpected findings and limits

The benchmark contains 833 tool error events across 499 attempts. Of those attempts, 249 ultimately pass after evaluator corrections. Intermediate errors must not be counted as 499 final operational failures.

Nine terminal failures are platform failures: seven internal Spark/metric-view errors and two timeouts. One is on pending task 31, leaving eight in the current main classification. Qwen accounts for 225 of the 233 execution failures; completion and answer correctness are separate dimensions.

The SQLite snapshot contains 2,637 of the 2,640 canonical case spans. Three final Qwen task 55 spans are missing; their terminal exception causes can still be classified from the report stacks. Sixty-seven attempts have scrubbed tool arguments. The 206 supplemental case spans are excluded from the canonical benchmark matrix.

This analysis does not rerun agents, modify evaluators, change scores, or update thesis figures. It accounts for every recorded failure, but it does not establish that every latent error in a passing answer has been found.

Review files
attempt_classifications.csv — all 2,640 attempts, category, secondary labels, confidence and trace IDs.
summary.json — complete counts and classification policy.
operational.json / interpretation.json / retrieval.json / analytical.json / visualization.json — optional underlying evidence.

Reproduce
From the repository root, run the existing virtual environment:
.venv/bin/python -m scripts.benchmark_failure_analysis.extract
.venv/bin/python -m scripts.benchmark_failure_analysis.operational
.venv/bin/python -m scripts.benchmark_failure_analysis.retrieval
.venv/bin/python -m scripts.benchmark_failure_analysis.interpretation
.venv/bin/python -m scripts.benchmark_failure_analysis.visualization
.venv/bin/python -m scripts.benchmark_failure_analysis.analytical
.venv/bin/python -m scripts.benchmark_failure_analysis.report

Each module accepts --output PATH. The extractor opens SQLite read-only. Scripts use local traces, reports and artifacts only; no network, model or platform calls are made. Detailed source evidence is separate from this report.
"""
    (output / "report.txt").write_text(report)
    write_json(
        output / "verification.json",
        {
            "attempts": len(ledger),
            "unique_attempts": len({row["case_span_id"] for row in ledger}),
            "all_failures_accounted_for": counts["unresolved"] == 0,
            "corrected_report_hashes_preserved": True,
            "original_report_hashes_preserved": True,
            "source_script_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    )
    print(dict(summary_counts))
    print(output / "report.txt")


if __name__ == "__main__":
    main()
