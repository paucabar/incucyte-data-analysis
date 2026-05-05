"""Plate geometry constants and well utilities.

Keep PLATE_ROWS, PLATE_COLS, and WELL_COUNTS in sync with
incucyte-portal/utils/constants.py.
"""
import re

PLATE_ROWS = {
    "SPL_6well":      list("AB"),
    "Nunc_48well":    list("ABCDEF"),
    "Greiner_96well": list("ABCDEFGH"),
}
PLATE_COLS = {
    "SPL_6well":      [str(c) for c in range(1, 4)],
    "Nunc_48well":    [str(c) for c in range(1, 9)],
    "Greiner_96well": [str(c) for c in range(1, 13)],
}
WELL_COUNTS = {"SPL_6well": 6, "Nunc_48well": 48, "Greiner_96well": 96}

_WELL_RE = re.compile(r"^([A-H])(\d{1,2})$")


def well_sort_key(well: str) -> tuple[int, int]:
    """Return (row_index, col_index) for row-major sorting of well labels."""
    m = _WELL_RE.match(well)
    if not m:
        raise ValueError(f"Invalid well label: {well!r}")
    return (ord(m.group(1)) - ord("A"), int(m.group(2)))
