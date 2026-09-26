from math import floor, isfinite, log10


def numbers_match(expected: float, actual: float, tolerance: float = 0.005) -> bool:
    """Compare finite numbers with an absolute tolerance scaled down below one."""
    if not isfinite(expected) or not isfinite(actual):
        return False
    if expected == actual:
        return True
    # Cap the scale at one; use a small fixed scale when the target is zero.
    scale = 10.0 ** min(0, floor(log10(abs(expected)))) if expected else 1e-6
    return abs(actual - expected) <= tolerance * scale
