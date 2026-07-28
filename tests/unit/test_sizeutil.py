"""Unit tests for gpuq.sizeutil — parse_size and format_size."""

import pytest

from gpuq.sizeutil import parse_size, format_size


# ---------------------------------------------------------------------------
# parse_size — valid inputs
# ---------------------------------------------------------------------------

class TestParseSizeValid:
    def test_gigabytes_integer(self):
        assert parse_size("8G") == 8 * 1024 ** 3

    def test_gigabytes_lowercase(self):
        assert parse_size("8g") == 8 * 1024 ** 3

    def test_megabytes(self):
        assert parse_size("512M") == 512 * 1024 ** 2

    def test_megabytes_lowercase(self):
        assert parse_size("512m") == 512 * 1024 ** 2

    def test_kilobytes(self):
        assert parse_size("1024K") == 1024 * 1024

    def test_terabytes(self):
        assert parse_size("2T") == 2 * 1024 ** 4

    def test_decimal_gigabytes(self):
        assert parse_size("1.5G") == int(1.5 * 1024 ** 3)

    def test_decimal_megabytes(self):
        assert parse_size("2.5M") == int(2.5 * 1024 ** 2)

    def test_bytes_suffix(self):
        assert parse_size("8B") == 8

    def test_bytes_suffix_lowercase(self):
        assert parse_size("8b") == 8

    def test_trailing_b(self):
        # "8GB" -> same as "8G"
        assert parse_size("8GB") == 8 * 1024 ** 3

    def test_trailing_ib(self):
        # "8GiB" -> same as "8G"
        assert parse_size("8GiB") == 8 * 1024 ** 3

    def test_megabytes_with_trailing_b(self):
        assert parse_size("512MB") == 512 * 1024 ** 2

    def test_kilobytes_with_trailing_ib(self):
        assert parse_size("8KiB") == 8 * 1024

    def test_internal_whitespace_accepted(self):
        # "8 G" should be accepted (strip internal spaces)
        assert parse_size("8 G") == 8 * 1024 ** 3

    def test_leading_trailing_whitespace(self):
        assert parse_size("  512M  ") == 512 * 1024 ** 2

    def test_gib_equals_g(self):
        # binary: 8GiB == 8G
        assert parse_size("8GiB") == parse_size("8G")

    def test_result_is_int(self):
        result = parse_size("1.5G")
        assert isinstance(result, int)

    def test_small_bytes(self):
        assert parse_size("1B") == 1

    def test_one_k(self):
        assert parse_size("1K") == 1024

    def test_one_m(self):
        assert parse_size("1M") == 1024 ** 2

    def test_one_g(self):
        assert parse_size("1G") == 1024 ** 3

    def test_one_t(self):
        assert parse_size("1T") == 1024 ** 4


# ---------------------------------------------------------------------------
# parse_size — invalid inputs (must raise ValueError)
# ---------------------------------------------------------------------------

class TestParseSizeInvalid:
    def test_bare_number_rejected(self):
        with pytest.raises(ValueError, match="invalid size"):
            parse_size("8000")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            parse_size("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            parse_size("   ")

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            parse_size("0G")

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            parse_size("-1G")

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            parse_size("abc")

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValueError):
            parse_size("8X")

    def test_no_unit_float_rejected(self):
        with pytest.raises(ValueError):
            parse_size("1.5")

    def test_none_like_input(self):
        # A non-string should raise
        with pytest.raises((ValueError, AttributeError, TypeError)):
            parse_size(None)  # type: ignore[arg-type]

    def test_zero_bytes_rejected(self):
        with pytest.raises(ValueError):
            parse_size("0B")


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_zero(self):
        assert format_size(0) == "0B"

    def test_bytes_below_k(self):
        assert format_size(512) == "512B"

    def test_exact_k(self):
        assert format_size(1024) == "1K"

    def test_exact_m(self):
        assert format_size(1024 ** 2) == "1M"

    def test_exact_g(self):
        assert format_size(1024 ** 3) == "1G"

    def test_exact_t(self):
        assert format_size(1024 ** 4) == "1T"

    def test_8g(self):
        assert format_size(8 * 1024 ** 3) == "8G"

    def test_512m(self):
        assert format_size(512 * 1024 ** 2) == "512M"

    def test_1_5g(self):
        assert format_size(int(1.5 * 1024 ** 3)) == "1.5G"

    def test_no_trailing_dot_zero(self):
        # Should be "8G" not "8.0G"
        result = format_size(8 * 1024 ** 3)
        assert "." not in result

    def test_2t(self):
        assert format_size(2 * 1024 ** 4) == "2T"

    def test_1_byte(self):
        assert format_size(1) == "1B"

    def test_1023_bytes(self):
        assert format_size(1023) == "1023B"

    def test_1k(self):
        assert format_size(1024) == "1K"


# ---------------------------------------------------------------------------
# Round-trip: parse_size -> format_size
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "8G", "512M", "1K", "2T", "1G",
    ])
    def test_roundtrip(self, text):
        """format_size(parse_size(x)) should reproduce x for clean values."""
        assert format_size(parse_size(text)) == text

    def test_decimal_roundtrip(self):
        # 1.5G parses then formats back to 1.5G
        assert format_size(parse_size("1.5G")) == "1.5G"
