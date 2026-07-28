"""Memory size parsing and formatting using binary multiples.

1 K = 1024 bytes
1 M = 1024**2 bytes
1 G = 1024**3 bytes
1 T = 1024**4 bytes
"""

import re
from typing import Union

# Binary unit multipliers
_UNITS = {
    "b": 1,
    "k": 1024,
    "m": 1024 ** 2,
    "g": 1024 ** 3,
    "t": 1024 ** 4,
}

# Regex: optional whitespace, number (int or float), optional whitespace,
# unit letter (B/K/M/G/T), optional trailing B or iB (case-insensitive).
_SIZE_RE = re.compile(
    r"""^\s*
        (?P<number>[0-9]+(?:\.[0-9]+)?)   # integer or decimal
        \s*
        (?P<unit>[bBkKmMgGtT])            # single-letter unit
        (?:i?[bB])?                       # optional trailing B or iB
        \s*$
    """,
    re.VERBOSE,
)


def parse_size(text: str) -> int:
    """Parse a human-readable memory size string into bytes.

    Accepted forms (case-insensitive suffix, optional trailing B/iB):
        "8G", "8g", "8GB", "8GiB", "1.5G", "512M", "1024K", "2T", "8B",
        "8 G"  (internal whitespace between number and unit is accepted).

    A bare number without a unit suffix (e.g. "8000") is rejected.
    Result is an int (floor for decimals). Must be > 0.

    Raises ValueError for any invalid input.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"invalid size {text!r}: use a unit like 8G, 512M, or 1.5G"
        )

    m = _SIZE_RE.match(text)
    if m is None:
        raise ValueError(
            f"invalid size {text!r}: use a unit like 8G, 512M, or 1.5G"
        )

    number_str = m.group("number")
    unit_char = m.group("unit").lower()

    try:
        value = float(number_str)
    except ValueError:
        raise ValueError(
            f"invalid size {text!r}: use a unit like 8G, 512M, or 1.5G"
        )

    if value <= 0:
        raise ValueError(
            f"invalid size {text!r}: size must be greater than zero"
        )

    multiplier = _UNITS[unit_char]
    result = int(value * multiplier)

    if result <= 0:
        raise ValueError(
            f"invalid size {text!r}: size must be greater than zero"
        )

    return result


def format_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses binary multiples (K=1024, M=1024**2, G=1024**3, T=1024**4).
    Picks the largest unit that yields a value >= 1.
    Up to one decimal place; trailing ".0" is stripped.

    Examples:
        8 * 1024**3        -> "8G"
        512 * 1024**2      -> "512M"
        1536 * 1024**2     -> "1.5G"
        512                -> "512B"
        0                  -> "0B"
    """
    if num_bytes == 0:
        return "0B"

    for unit, multiplier in (("T", 1024 ** 4), ("G", 1024 ** 3), ("M", 1024 ** 2), ("K", 1024)):
        if num_bytes >= multiplier:
            value = num_bytes / multiplier
            # Format with up to 1 decimal, strip trailing .0
            formatted = f"{value:.1f}"
            if formatted.endswith(".0"):
                formatted = formatted[:-2]
            return f"{formatted}{unit}"

    return f"{num_bytes}B"
