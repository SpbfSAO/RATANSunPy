from .context import ratansunpy
from ratansunpy.time import check_equal_time, get_time_format
from ratansunpy.time import TimeFormats
import astropy.units as u
import pytest
from astropy.time import Time
import pytest


@pytest.fixture
def times_different():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    t1 = Time('2024-08-06T00:00:00.000000000')
    t2 = Time('2024-08-06T00:00:01.000000000')
    return t1, t2


@pytest.fixture
def times_identical():
    """
    Fixture that provides two identical Time objects.
    """
    t1 = Time('2024-08-06T00:00:00.000000000')
    t2 = Time('2024-08-06T00:00:00.000000000')
    return t1, t2


@pytest.fixture
def times_identical_diff_format():
    """
    Fixture that provides two identical Time objects.
    """
    t1 = Time('2024-08-06T00:00:00.000000000')
    t2 = Time('2024-08-06T00:00:00')
    #t2 = Time('2024.08.06_00:00:00_TAI')
    return t1, t2

@pytest.fixture
def common_time_formats():
    formats = TimeFormats().common_formats
    examples = TimeFormats().examples
    return formats, examples

@pytest.fixture
def times_edge_case():
    """
    Fixture that provides two identical Time objects with potentially different scales.
    """
    t1 = Time('2024-08-06T00:00:00.000000000')
    t2 = Time('2024-08-06T00:00:00.000000001')
    return t1, t2


def test_time_formats():
    common_time_formats = TimeFormats()
    assert len(common_time_formats.common_formats) == 30
    assert "%Y-%m-%dT%H:%M:%S.%f" in common_time_formats.common_formats


class TestCheckEqualTime:
    """
    Test cases for the `check_equal_time` function.
    """

    def test_check_equal_time_equal(self, times_identical):
        """
        Test case where two times are within 1 nanosecond of each other.
        """
        t1, t2 = times_identical
        assert check_equal_time(t1, t2) is True

    def test_check_equal_time_not_equal(self, times_different):
        """
        Test case where two times are more than 1 nanosecond apart.
        """
        t1, t2 = times_different
        assert check_equal_time(t1, t2) is False

    def test_check_equal_time_identical(self, times_identical_diff_format):
        """
        Test case where two times are exactly the same.
        """
        t1, t2 = times_identical_diff_format
        assert check_equal_time(t1, t2) is True

    def test_check_equal_time_edge_case(self, times_edge_case):
        """
        Test case where the times are identical but may involve different scales or formats.
        """
        t1, t2 = times_edge_case
        assert check_equal_time(t1, t2) is True


#TODO test not passed, need to check regexp
class TestGetTimeFormat:
    def test_diff_formats(self, common_time_formats):
        formats, examples = common_time_formats
        err_list = []
        for fmt, exp in zip(formats, examples):
            inf_fmt = get_time_format(exp)
            if inf_fmt != fmt:
                err_list.append((exp, inf_fmt, fmt))
        assert len(err_list) == 0, f"Mismatched formats: {err_list}"