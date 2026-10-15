"""Match table contents with one-to-one column and whole-row assignments.

Column labels do not participate in matching. Try exact column vectors first,
then ordered assignments, and search ambiguous assignments only when necessary.
"""

from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from math import floor, isfinite, log10
from numbers import Real
from typing import NamedTuple

import pandas as pd

from ._table_values import _normalized_rows, _rounded_rows, _rows_match


class MatchModes(NamedTuple):
    ordered: bool
    unordered: bool


def _distinct_matches(matches, target_count):
    """Assign distinct targets, reassigning choices when possible matches overlap."""
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
            return None

        # Flip the alternating path to free a target for this candidate row.
        while end is not None:
            candidate, target = end
            owners[target] = candidate
            end = parents[candidate]
    assignment = [None] * len(matches)
    for target, candidate in enumerate(owners):
        if candidate is not None:
            assignment[candidate] = target
    return assignment


def _ordered_rows_match(target_rows, candidate_rows, tolerance):
    if len(target_rows) == len(candidate_rows):
        return all(
            _rows_match(target, candidate, tolerance)
            for target, candidate in zip(target_rows, candidate_rows, strict=True)
        )
    # An ordered partial match must be a subsequence of complete reference rows.
    remaining = iter(target_rows)
    return all(
        any(_rows_match(target, candidate, tolerance) for target in remaining)
        for candidate in candidate_rows
    )


def _match_rows(target_rows, candidate_rows, tolerance):
    if _ordered_rows_match(target_rows, candidate_rows, tolerance):
        return MatchModes(True, True)

    # Column searches can reject disjoint numeric ranges and compare sorted values
    # before constructing the more expensive one-to-one row matching graph.
    if target_rows and candidate_rows and len(target_rows[0]) == 1:
        expected = [row[0] for row in target_rows]
        actual = [row[0] for row in candidate_rows]
        if all(
            isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
            for value in expected + actual
        ):
            error = tolerance * max(
                10.0 ** min(0, floor(log10(abs(value)))) if value else 1e-6
                for value in expected
            )
            if (
                min(actual) < min(expected) - error
                or max(actual) > max(expected) + error
            ):
                return MatchModes(False, False)
            if len(expected) == len(actual) and (
                min(actual) > min(expected) + error
                or max(actual) < max(expected) - error
            ):
                return MatchModes(False, False)
            if _ordered_rows_match(
                sorted(target_rows), sorted(candidate_rows), tolerance
            ):
                return MatchModes(False, True)

    matches = [
        [
            index
            for index, target in enumerate(target_rows)
            if _rows_match(target, candidate, tolerance)
        ]
        for candidate in candidate_rows
    ]
    return MatchModes(False, _distinct_matches(matches, len(target_rows)) is not None)


def _compare_rows(target_rows, candidate_rows, partial, tolerance, rounding):
    modes = _match_rows(target_rows, candidate_rows, tolerance)
    if modes.ordered or partial or rounding is None or tolerance == 0:
        return modes
    rounded = _rounded_rows(target_rows, candidate_rows, rounding)
    if rounded is None:
        return modes
    rounded_target, rounded_candidate = rounded
    rounded_modes = _match_rows(rounded_target, rounded_candidate, tolerance)
    return MatchModes(rounded_modes.ordered, modes.unordered or rounded_modes.unordered)


def _column_value_kinds(values):
    return Counter(
        Real if isinstance(value, Real) and not isinstance(value, bool) else type(value)
        for value in values
    )


def _column_signature(column):
    # Python equates True and 1; the evaluator deliberately keeps them distinct.
    return tuple(
        (bool, value) if isinstance(value, bool) else value for value in column
    )


def _exact_column_mapping(target_columns, candidate_columns):
    by_contents = defaultdict(deque)
    for index, column in enumerate(candidate_columns):
        by_contents[_column_signature(column)].append(index)
    mapping = []
    for column in target_columns:
        matches = by_contents.get(_column_signature(column))
        if not matches:
            return None
        mapping.append(matches.popleft())
    return mapping


def _column_mappings(
    target_rows, candidate_rows, column_count, partial, tolerance, rounding
) -> Iterator[tuple[list[int], MatchModes]]:
    """Yield valid column assignments with their already-computed row matches."""
    # Avoid building a search when the values already align, including tolerance.
    if (
        candidate_rows
        and len(candidate_rows[0]) == column_count
        and _ordered_rows_match(target_rows, candidate_rows, tolerance)
    ):
        yield list(range(column_count)), MatchModes(True, True)
        return

    target_columns = list(zip(*target_rows)) if target_rows else [()] * column_count
    candidate_columns = (
        list(zip(*candidate_rows)) if candidate_rows else [()] * column_count
    )

    def compare(mapping):
        expected = [tuple(row[i] for i in mapping) for row in target_rows]
        actual = [tuple(row[j] for j in mapping.values()) for row in candidate_rows]
        return _compare_rows(expected, actual, partial, tolerance, rounding)

    if column_count == 0:
        yield [], compare({})
        return

    if not partial:
        mapping = _exact_column_mapping(target_columns, candidate_columns)
        if mapping is not None:
            modes = compare(dict(enumerate(mapping)))
            if modes.ordered:
                yield mapping, modes
                return

    target_kinds = [_column_value_kinds(column) for column in target_columns]
    candidate_kinds = [_column_value_kinds(column) for column in candidate_columns]
    choices, ordered_choices = [], []
    for i, column in enumerate(target_columns):
        expected = [(value,) for value in column]
        matches = []
        for j, candidate_column in enumerate(candidate_columns):
            if not candidate_kinds[j] <= target_kinds[i]:
                continue
            if Real not in target_kinds[i] and not Counter(candidate_column) <= Counter(
                column
            ):
                continue
            actual = [(value,) for value in candidate_column]
            modes = _compare_rows(expected, actual, partial, tolerance, rounding)
            if modes.unordered:
                matches.append((not modes.ordered, j))
        if not matches:
            return
        choices.append([j for _, j in sorted(matches)])
        ordered_choices.append([j for unordered, j in matches if not unordered])

    ordered_mapping = _distinct_matches(ordered_choices, len(candidate_columns))
    if ordered_mapping is not None:
        modes = compare(dict(enumerate(ordered_mapping)))
        if modes.ordered:
            yield ordered_mapping, modes
            return
    if _distinct_matches(choices, len(candidate_columns)) is None:
        return

    # Start with the least ambiguous column. Every partial assignment must also
    # match whole rows, so independently correct columns cannot hide swapped cells.
    search_order = sorted(range(column_count), key=lambda i: len(choices[i]))
    candidate_signatures = [_column_signature(column) for column in candidate_columns]

    def search(mapping):
        i = search_order[len(mapping)]
        seen = set()
        for j in choices[i]:
            signature = candidate_signatures[j]
            if j in mapping.values() or signature in seen:
                continue
            seen.add(signature)
            selected = {**mapping, i: j}
            modes = compare(selected)
            if not modes.unordered:
                continue
            if len(selected) == column_count:
                yield [selected[i] for i in range(column_count)], modes
            else:
                yield from search(selected)

    for result in search({}):
        yield result
        # Without an ordered column assignment, no whole-row assignment can be
        # ordered. The first valid unordered result is therefore sufficient.
        if ordered_mapping is None:
            return


def _compatible_shape(
    target_table: pd.DataFrame, candidate_table: pd.DataFrame
) -> bool:
    return (
        len(candidate_table.columns) >= len(target_table.columns)
        and target_table.columns.is_unique
        and candidate_table.columns.is_unique
        and len(candidate_table) <= len(target_table)
    )


def _column_positions(target_table: pd.DataFrame, candidate_table: pd.DataFrame):
    """Expose a content-based assignment for the existing benchmark audit scripts."""
    if not _compatible_shape(target_table, candidate_table):
        return None
    for mapping, _ in _column_mappings(
        _normalized_rows(target_table),
        _normalized_rows(candidate_table),
        len(target_table.columns),
        len(candidate_table) < len(target_table),
        0.005,
        None,
    ):
        return mapping
    return None


def table_match_modes(
    target_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    partial: bool = False,
    tolerance: float = 0.005,
    minimum_significant_digits: int | None = None,
) -> MatchModes:
    """Return whether complete rows match in order and ignoring order."""
    if not _compatible_shape(target_table, candidate_table):
        return MatchModes(False, False)
    if not partial and len(candidate_table) != len(target_table):
        return MatchModes(False, False)
    unordered = False
    for _, modes in _column_mappings(
        _normalized_rows(target_table),
        _normalized_rows(candidate_table),
        len(target_table.columns),
        partial,
        tolerance,
        minimum_significant_digits,
    ):
        if modes.ordered:
            return modes
        unordered |= modes.unordered
    return MatchModes(False, unordered)
