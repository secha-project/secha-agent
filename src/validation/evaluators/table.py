import re
from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from math import isfinite
from numbers import Real
from pathlib import Path

import pandas as pd
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from ..eval_models import BenchmarkOutput, CaseMetadata, TableOutput
from ._numeric import numbers_match

_MISSING = object()
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
    r"(?:Z|[+-]\d{2}:\d{2}| UTC)?)?"
)
_EXPLICIT_PRECISION = re.compile(
    r"\bunrounded\b|\b(?:full|original)\s+precision\b"
    r"|\b(?:without|no)\s+rounding\b|\b(?:do not|don't)\s+round\b"
    r"|\b(?:decimal|significant)\s+(?:places?|figures?|digits?)\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)[ -]decimals?\b",
    re.IGNORECASE,
)

# Field identities preserve units and phase order. They never depend on values:
# searching for whichever permutation passes would hide incorrectly labeled data.
_COLUMN_ALIAS_GROUPS = (
    ("timestamp", "timestamp_utc", "time_utc", "measurement_time_utc"),
    ("pw", "power_w", "active_power_w", "measurement_power_w"),
    ("device_name", "device_identifier", "solar_power_plant"),
    ("description", "catalog_description"),
    ("freq", "frequency_hz"),
    ("start_utc", "first_timestamp", "first_ts"),
    ("end_utc", "last_timestamp", "last_ts"),
    ("duration_seconds", "duration_s", "elapsed_duration_s"),
    (
        "mean_opposing_w",
        "mean_abs_w",
        "mean_mag_w",
        "mean_nf_magnitude_w",
        "mean_nonfundamental_mag_w",
        "mean_nonfundamental_w",
    ),
    (
        "peak_opposing_w",
        "peak_abs_w",
        "peak_mag_w",
        "peak_nf_magnitude_w",
        "peak_nonfundamental_mag_w",
        "peak_nonfundamental_w",
    ),
    ("mean_absolute_discrepancy_va", "mean_abs_discrepancy_va"),
    ("measurement_date", "utc_date"),
    ("morning_mean_power_kw", "mean_power_06_10_kw"),
    ("evening_mean_power_kw", "mean_power_16_20_kw"),
    ("absolute_difference_kw", "abs_diff_kw"),
    ("hour_start", "hour_utc"),
    ("mean_positive_charging_power_kw", "mean_positive_power_kw"),
    ("spot_price_eur_per_mwh", "fi_spot_price_eur_mwh"),
    ("l1", "phase_one"),
    ("l2", "phase_two"),
    ("l3", "phase_three"),
)
_COLUMN_ALIASES = {alias: group[0] for group in _COLUMN_ALIAS_GROUPS for alias in group}


def _semantic_column_name(name):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if normalized in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[normalized]
    if match := re.fullmatch(r"u(?:l)?([123])([123]?)v", normalized):
        first, second = match.groups()
        phase = f"l{first}" + (f"_l{second}" if second else "")
        return f"voltage_{phase}_v"
    if match := re.fullmatch(
        r"(?:(?:phase|supply)_)?voltage_(l[123](?:_l[123])?)_v", normalized
    ):
        return f"voltage_{match[1]}_v"
    if match := re.fullmatch(r"phase_(l[123](?:_l[123])?)_voltage_v", normalized):
        return f"voltage_{match[1]}_v"
    if match := re.fullmatch(r"tdul([123])", normalized):
        return f"voltage_distortion_l{match[1]}_pct"
    if match := re.fullmatch(r"(?:total_)?voltage_distortion_l([123])_pct", normalized):
        return f"voltage_distortion_l{match[1]}_pct"
    return None


def _normalize_value(value):
    if pd.isna(value):
        return _MISSING
    if isinstance(value, str) and _ISO_TIMESTAMP.fullmatch(value):
        try:
            if value.endswith(" UTC"):
                value = value[:-4] + "Z"
            timestamp = pd.Timestamp(value)
            # Reference CSVs store timezone-free timestamps in UTC.
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")
        except ValueError:
            pass
    return value


def _values_match(expected, actual, tolerance):
    if expected is _MISSING or actual is _MISSING:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, Real) and isinstance(actual, Real):
        return numbers_match(expected, actual, tolerance)
    return expected == actual


def _column_positions(target_table, candidate_table):
    """Match exact names, then explicit field aliases, then unknown aliases in order."""
    if (
        len(candidate_table.columns) < len(target_table.columns)
        or not target_table.columns.is_unique
        or not candidate_table.columns.is_unique
    ):
        return None

    positions = {name: index for index, name in enumerate(candidate_table.columns)}
    matched = [positions.get(name) for name in target_table.columns]
    used = {index for index in matched if index is not None}
    target_keys = [_semantic_column_name(name) for name in target_table.columns]
    candidate_keys = [_semantic_column_name(name) for name in candidate_table.columns]
    for target_index, key in enumerate(target_keys):
        if matched[target_index] is not None or key is None:
            continue
        aliases = [
            index
            for index, candidate_key in enumerate(candidate_keys)
            if index not in used and key == candidate_key
        ]
        if len(aliases) > 1:
            return None
        if aliases:
            matched[target_index] = aliases[0]
            used.add(aliases[0])
    remaining = iter(index for index in range(len(positions)) if index not in used)
    for target_index, index in enumerate(matched):
        if index is not None:
            continue
        index = next(remaining)
        if (
            target_keys[target_index] is not None
            and candidate_keys[index] is not None
            and target_keys[target_index] != candidate_keys[index]
        ):
            return None
        matched[target_index] = index
    return matched


def _rows_match(expected, actual, tolerance):
    return all(
        _values_match(target, candidate, tolerance)
        for target, candidate in zip(expected, actual, strict=True)
    )


def _has_distinct_row_matches(matches, target_count):
    """Find one-to-one matches, reassigning earlier choices when tolerances overlap."""
    owners = [None] * target_count
    for start in range(len(matches)):
        parents = {start: None}
        pending = [start]
        end = None
        for candidate in pending:
            for target in matches[candidate]:
                owner = owners[target]
                if owner is None:
                    end = (candidate, target)
                    break
                if owner not in parents:
                    parents[owner] = (candidate, target)
                    pending.append(owner)
            if end is not None:
                break
        if end is None:
            return False

        # Flip the alternating path to free a target for this candidate row.
        while end is not None:
            candidate, target = end
            owners[target] = candidate
            end = parents[candidate]
    return True


def _match_rows(target_rows, candidate_rows, tolerance):
    # An ordered partial match must be a subsequence of complete reference rows.
    remaining = iter(target_rows)
    if all(
        any(_rows_match(target, candidate, tolerance) for target in remaining)
        for candidate in candidate_rows
    ):
        return True, True

    matches = [
        [
            index
            for index, target in enumerate(target_rows)
            if _rows_match(target, candidate, tolerance)
        ]
        for candidate in candidate_rows
    ]
    return False, _has_distinct_row_matches(matches, len(target_rows))


def _rounded_column(expected, actual, minimum_significant_digits):
    """Find one rounding rule that explains every value in a measured column."""
    values = [value for value in expected + actual if value is not _MISSING]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
        for value in values
    ):
        return None
    # Integer-valued reference columns include counts, IDs, and whole durations.
    if all(float(value).is_integer() for value in expected if value is not _MISSING):
        return None
    if expected.count(_MISSING) != actual.count(_MISSING):
        return None
    target = [
        Decimal(str(value)) if value is not _MISSING else _MISSING for value in expected
    ]
    candidate = [
        Decimal(str(value)) if value is not _MISSING else _MISSING for value in actual
    ]
    candidate_counts = Counter(candidate)
    rules = [("decimal_places", places) for places in range(16)] + [
        ("significant_figures", digits)
        for digits in range(minimum_significant_digits, 16)
    ]
    for kind, precision in rules:
        rounded = []
        for value in target:
            if value is _MISSING:
                rounded.append(value)
                continue
            places = (
                precision
                if kind == "decimal_places"
                else precision - value.adjusted() - 1
            )
            try:
                result = value.quantize(
                    Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
                )
            except InvalidOperation:
                break
            if result != value and (
                result == 0
                or value.adjusted() + places + 1 < minimum_significant_digits
            ):
                break
            rounded.append(result)
        else:
            if rounded != target and Counter(rounded) == candidate_counts:
                return rounded, candidate
    return None


def _rounded_rows(target_rows, candidate_rows, minimum_significant_digits):
    """Round fixed columns, then let whole-row matching verify their associations."""
    if not target_rows or len(target_rows) != len(candidate_rows):
        return None
    target = [list(row) for row in target_rows]
    candidate = [list(row) for row in candidate_rows]
    changed = False
    for index in range(len(target[0])):
        column = _rounded_column(
            [row[index] for row in target],
            [row[index] for row in candidate],
            minimum_significant_digits,
        )
        if column is None:
            continue
        # Decimal cells compare exactly; no extra tolerance is added after rounding.
        for rows, values in zip((target, candidate), column, strict=True):
            for row, value in zip(rows, values, strict=True):
                row[index] = value
        changed = True
    return (target, candidate) if changed else None


def table_match_modes(
    target_table,
    candidate_table,
    partial=False,
    tolerance=0.005,
    minimum_significant_digits=None,
):
    """Return whether whole rows match in order and ignoring order, respectively."""
    positions = _column_positions(target_table, candidate_table)
    if positions is None or len(candidate_table) > len(target_table):
        return False, False
    if not partial and len(candidate_table) != len(target_table):
        return False, False

    target_rows = [
        tuple(_normalize_value(value) for value in row)
        for row in target_table.itertuples(index=False, name=None)
    ]
    candidate_rows = [
        tuple(_normalize_value(value) for value in row)
        for row in candidate_table.iloc[:, positions].itertuples(index=False, name=None)
    ]
    modes = _match_rows(target_rows, candidate_rows, tolerance)
    if modes[1] or partial or minimum_significant_digits is None or tolerance == 0:
        return modes
    rounded = _rounded_rows(target_rows, candidate_rows, minimum_significant_digits)
    return _match_rows(*rounded, tolerance) if rounded is not None else modes


def classify_table_match(
    target_table,
    candidate_table,
    min_coverage=0.5,
    tolerance=0.005,
    minimum_significant_digits=None,
):
    target_rows = len(target_table)
    candidate_rows = len(candidate_table)

    if target_rows == 0:
        if candidate_rows != 0:
            return "no_match", None

        ordered, unordered = table_match_modes(
            target_table,
            candidate_table,
            partial=False,
            tolerance=tolerance,
            minimum_significant_digits=minimum_significant_digits,
        )

        if ordered:
            return "exact_match", 1.0
        if unordered:
            return "match_up_to_ordering", 1.0
        return "no_match", None

    coverage = candidate_rows / target_rows

    if candidate_rows > target_rows:
        return "no_match", coverage

    partial = candidate_rows < target_rows

    if partial and coverage < min_coverage:
        return "no_match", coverage

    ordered, unordered = table_match_modes(
        target_table,
        candidate_table,
        partial=partial,
        tolerance=tolerance,
        minimum_significant_digits=minimum_significant_digits,
    )

    if ordered:
        match_type = "partial_exact_match" if partial else "exact_match"
        return match_type, coverage

    if unordered:
        match_type = (
            "partial_match_up_to_ordering" if partial else "match_up_to_ordering"
        )
        return match_type, coverage

    return "no_match", coverage


def _table_outputs(expected_output):
    if not isinstance(expected_output, list):
        return []

    return [item for item in expected_output if isinstance(item, TableOutput)]


@dataclass
class ValidateTableIncluded(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that the agent selected a table to return to the user."""

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        if not _table_outputs(ctx.expected_output):
            return {}

        return "csv_path" in ctx.attributes


@dataclass
class ValidateTableValues(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Compare the selected table with every expected table."""

    artifact_base_path: Path = Path()
    min_coverage: float = 0.5
    tolerance: float = 0.005
    minimum_significant_digits: int = 2

    def __post_init__(self):
        if not 2 <= self.minimum_significant_digits <= 15:
            raise ValueError("Table rounding requires 2 to 15 significant digits")

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        table_outputs = _table_outputs(ctx.expected_output)
        if not table_outputs:
            return {}

        csv_path = ctx.attributes.get("csv_path")
        if not isinstance(csv_path, (str, Path)):
            return {
                "ValidateTableValues": False,
                "TableMatchType": "no_match",
            }

        candidate_table = pd.read_csv(csv_path)
        overall_match_type = "exact_match"
        inputs = getattr(ctx, "inputs", "")
        rounding = (
            None
            if isinstance(inputs, str) and _EXPLICIT_PRECISION.search(inputs)
            else self.minimum_significant_digits
        )

        for table_output in table_outputs:
            target_path = Path(table_output.table_path)
            if not target_path.is_absolute():
                target_path = self.artifact_base_path / target_path

            target_table = pd.read_csv(target_path)
            match_type, _ = classify_table_match(
                target_table,
                candidate_table,
                min_coverage=self.min_coverage,
                tolerance=self.tolerance,
                minimum_significant_digits=rounding,
            )
            if match_type != "exact_match" and not (
                match_type == "match_up_to_ordering" and not table_output.ordered
            ):
                return {
                    "ValidateTableValues": False,
                    "TableMatchType": match_type,
                }

            if match_type == "match_up_to_ordering":
                overall_match_type = match_type

        return {
            "ValidateTableValues": True,
            "TableMatchType": overall_match_type,
        }
