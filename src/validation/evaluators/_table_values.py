"""Cell normalization and precision rules for table comparisons.

Normalize timestamps and missing cells before matching. Rounded numeric columns
use Decimal values so no second floating-point tolerance is applied to them.
"""

import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from math import isfinite
from numbers import Real

import pandas as pd

from ._numeric import numbers_match

_MISSING = object()
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
    r"(?:Z|[+-]\d{2}:\d{2}| UTC)?)?"
)


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


def _rows_match(expected, actual, tolerance):
    return all(
        _values_match(target, candidate, tolerance)
        for target, candidate in zip(expected, actual, strict=True)
    )


def _normalized_rows(table):
    return [
        tuple(_normalize_value(value) for value in row)
        for row in table.itertuples(index=False, name=None)
    ]


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
            if result not in candidate_counts or (
                result != value
                and (
                    result == 0
                    or value.adjusted() + places + 1 < minimum_significant_digits
                )
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
