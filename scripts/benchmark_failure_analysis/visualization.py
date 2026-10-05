"""Extract saved judge verdicts and check actual chart data and encodings."""

import json
import math
import re
from collections import Counter

from .common import (
    REFERENCE,
    arguments,
    artifact,
    finding,
    identity,
    read_evidence,
    save_findings,
)


def rows(spec):
    return [row for data in spec.get("datasets", {}).values() for row in data]


def layers(spec):
    yield spec
    for key in ("layer", "concat", "hconcat", "vconcat"):
        for child in spec.get(key, []):
            yield from layers(child)
    if isinstance(spec.get("spec"), dict):
        yield from layers(spec["spec"])


def data_check(task, spec):
    actual = rows(spec)
    checks = {}
    if task not in {"task_26", "task_29", "task_42", "task_44"}:
        return checks
    reference = rows(
        json.loads((REFERENCE / f"artifacts/{task}/visualization.json").read_text())
    )
    if task == "task_26":
        expected = {r["__ggsql_aes_pos1__"]: r["__ggsql_aes_pos2__"] for r in reference}
        observed = {}
        for row in actual:
            value, count = row.get("__ggsql_aes_pos1__"), row.get("__ggsql_aes_pos2__")
            if isinstance(count, str) and isinstance(value, (int, float)):
                value, count = count, value - row.get("__ggsql_aes_pos1end__", 0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                match = re.match(r"\s*\[?\s*(-?\d+)", str(value))
                value = float(match[1]) if match else None
            if value is not None and isinstance(count, (int, float)):
                observed[value] = count - row.get("__ggsql_aes_pos2end__", 0)
        checks = {
            "all_bin_counts_match": observed == expected,
            "expected_counts": expected,
            "actual_counts": observed,
        }
    elif task == "task_29":
        expected = {
            (float(r["__ggsql_aes_pos1__"]), r["__ggsql_aes_fill__"]): r[
                "__ggsql_aes_pos2__"
            ]
            for r in reference
        }
        observed = {}
        for row in actual:
            phase = row.get("__ggsql_aes_fill__", row.get("__ggsql_aes_stroke__"))
            if phase not in {"L1", "L2", "L3"}:
                continue
            try:
                order = float(row.get("__ggsql_aes_pos1__"))
            except (TypeError, ValueError):
                continue
            if isinstance(row.get("__ggsql_aes_pos2__"), (int, float)):
                observed[order, phase] = row["__ggsql_aes_pos2__"] - row.get(
                    "__ggsql_aes_pos2end__", 0
                )
        checks = {
            "all_57_phase_means_match_at_four_decimals": len(observed) == 57
            and all(
                key in expected
                and math.isclose(value, expected[key], rel_tol=0, abs_tol=5.1e-5)
                for key, value in observed.items()
            ),
            "phase_values": len(observed),
            "nonzero_stack_baselines": sum(
                abs(r.get("__ggsql_aes_pos2end__", 0)) > 1e-12 for r in actual
            ),
        }
    elif task == "task_42":

        def series(label):
            text = str(label).lower()
            if "tenant" in text:
                return "tenants"
            if any(term in text for term in ("maintenance", "shared", "services")):
                return "maintenance"
            return None

        def values(data):
            result = {}
            for row in data:
                group = series(
                    row.get("__ggsql_aes_fill__", row.get("__ggsql_aes_stroke__"))
                )
                time = row.get("__ggsql_aes_pos1__")
                value = row.get("__ggsql_aes_pos2__")
                if group and isinstance(value, (int, float)):
                    result[time, group] = value - row.get("__ggsql_aes_pos2end__", 0)
            return result

        expected, observed = values(reference), values(actual)
        complete = observed.keys() == expected.keys()
        checks = {
            "labelled_series_complete": complete,
            "underlying_series_values_match": complete
            and all(
                abs(observed[key] - value) <= 0.0006 for key, value in expected.items()
            ),
            "tenant_series_halved": complete
            and all(
                abs(observed[key] - value * (0.5 if key[1] == "tenants" else 1))
                <= 0.0006
                for key, value in expected.items()
            ),
            "nonzero_stack_baselines": sum(
                abs(r.get("__ggsql_aes_pos2end__", 0)) > 1e-12 for r in actual
            ),
        }
    else:

        def hour(value):
            if isinstance(value, (int, float)) and 0 <= value <= 23:
                return f"{int(value):02d}:00"
            text = str(value)
            match = re.fullmatch(r"(\d{1,2})(?::00)?", text)
            if match:
                return f"{int(match[1]):02d}:00"
            match = re.search(r"[T ](\d{2}):\d{2}", text)
            return f"{match[1]}:00" if match else None

        def values(data):
            return {
                hour(row.get("__ggsql_aes_pos1__")): row["__ggsql_aes_pos2__"]
                - row.get("__ggsql_aes_pos2end__", 0)
                for row in data
                if hour(row.get("__ggsql_aes_pos1__")) is not None
                and isinstance(row.get("__ggsql_aes_pos2__"), (int, float))
            }

        expected, observed = values(reference), values(actual)
        complete = expected.keys() == observed.keys()
        equivalent = complete and any(
            all(
                abs(observed[key] - round(value, precision)) <= 0.0006
                for key, value in expected.items()
            )
            for precision in range(5)
        )
        checks = {
            "hours_complete": complete,
            "hourly_percentages_match_allowing_rounding": equivalent,
            "hourly_differences": {
                key: observed[key] - value
                for key, value in expected.items()
                if key in observed
            },
        }
    return checks


def main():
    output = arguments(__doc__).output
    findings, charts, rationales = [], [], []
    for attempt in read_evidence(output):
        if not any("image_path" in expected for expected in attempt["expected_output"]):
            continue
        assertions = {
            name: value
            for name, value in attempt.get("assertions", {}).items()
            if not value["value"]
        }
        spec_path = artifact(attempt, "visualization_spec_path")
        spec = (
            json.loads(spec_path.read_text())
            if spec_path and spec_path.is_file()
            else None
        )
        checks = data_check(attempt["task"], spec) if spec else {}
        bandwidth = bool(spec) and any(
            isinstance(layer.get("mark"), dict)
            and layer["mark"].get("type") == "bar"
            and "bandwidth" in str(layer["mark"].get("width"))
            and layer.get("encoding", {}).get("x", {}).get("type")
            in {"quantitative", "temporal"}
            for layer in layers(spec)
        )
        x_offset = bool(spec) and any(
            "xOffset" in layer.get("encoding", {}) for layer in layers(spec)
        )
        chart = {
            **identity(attempt),
            "spec": str(spec_path) if spec_path else None,
            "image": str(artifact(attempt, "image_path")),
            "checks": checks,
            "continuous_axis_bandwidth_bar": bandwidth,
            "has_x_offset": x_offset,
            "failed_assertions": assertions,
        }
        charts.append(chart)
        if attempt["passed"] or attempt["status"] == "failed":
            continue
        rationales.append(chart)
        findings.append(
            finding(
                attempt,
                "V0",
                "visualization",
                "Saved image evaluation rejects the completed plot.",
                chart,
                "saved judge verdict; exact causal checks listed separately",
            )
        )
        if bandwidth:
            findings.append(
                finding(
                    attempt,
                    "V1",
                    "visualization",
                    "Bar width uses categorical bandwidth on a continuous axis.",
                    {
                        "spec": str(spec_path),
                        "image": chart["image"],
                        "data_checks": checks,
                    },
                )
            )
        if (
            attempt["task"] == "task_29"
            and checks.get("all_57_phase_means_match_at_four_decimals")
            and not x_offset
        ):
            findings.append(
                finding(
                    attempt,
                    "V2",
                    "visualization",
                    "Correct phase means are stacked or overlaid instead of grouped side by side.",
                    {"spec": str(spec_path), **checks, "has_x_offset": x_offset},
                )
            )
        if assertions and all(
            "Aesthetic" in name or "aesthetic" in name for name in assertions
        ):
            findings.append(
                finding(
                    attempt,
                    "V3",
                    "visualization",
                    "Only the aesthetics assertion fails; do not infer incorrect calculations.",
                    {"failed_assertions": assertions, "image": chart["image"]},
                    "saved judge verdict",
                )
            )
        reason = " ".join(
            str(value.get("reason") or "") for value in assertions.values()
        ).lower()
        for rule, condition, explanation in (
            (
                "V4",
                "weekday" in reason or "alphabetical" in reason,
                "Saved judge rationale identifies misleading category/weekday order.",
            ),
            (
                "V5",
                "legend" in reason
                and any(
                    word in reason
                    for word in ("truncat", "indistinguish", "clip", "missing", "omit")
                ),
                "Saved judge rationale identifies clipped, ambiguous or missing legend entries.",
            ),
            (
                "V6",
                any(
                    word in reason
                    for word in (
                        "overlap",
                        "stacked",
                        "stacking",
                        "side-by-side",
                        "side by side",
                    )
                ),
                "Saved judge rationale identifies incorrect series positioning or stacking.",
            ),
        ):
            if condition:
                findings.append(
                    finding(
                        attempt,
                        rule,
                        "visualization",
                        explanation,
                        {"failed_assertions": assertions, "image": chart["image"]},
                        "saved judge rationale; keyword extraction",
                    )
                )
    save_findings(
        output,
        "visualization",
        findings,
        {
            "charts": charts,
            "rejections": rationales,
            "outcomes": dict(
                Counter("pass" if c["passed"] else c["status"] for c in charts)
            ),
            "note": "Original image verdicts retained; no new judge calls. V0 does not prove an analytical error.",
        },
    )
    print(
        "chart outcomes",
        dict(Counter("pass" if c["passed"] else c["status"] for c in charts)),
    )


if __name__ == "__main__":
    main()
