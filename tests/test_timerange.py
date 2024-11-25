import pytest
from datetime import timedelta
from astropy.time import Time, TimeDelta
import astropy.units as u
from ratansunpy.time import parse_time
from ratansunpy.time import TimeRange


@pytest.fixture
def time_range_example():
    start_time = Time("2023-01-01 00:00:00")
    end_time = Time("2023-01-02 00:00:00")
    delta = timedelta(days=1)
    return start_time, end_time, delta

@pytest.fixture
def time_tuple_str():
    start = "2023-01-01 00:00:00"
    end = "2023-01-02 00:00:00"
    return (start, end)

class TestTimeRange:

    def test_timerange_initialization_with_times(self,
                                                 time_range_example):
        start_time, end_time, delta = time_range_example
        timerange = TimeRange(start_time, end_time)

        assert timerange.start == start_time
        assert timerange.end == end_time
        assert timerange.days == 1
        assert timerange.hours == 24
        assert timerange.minutes == 1440
        assert timerange.seconds == 86400

    def test_timerange_initialization_with_timedelta(self,
                                                     time_range_example):
        start_time, end_time, delta = time_range_example

        timerange = TimeRange(start_time, delta)

        assert timerange.start == start_time
        assert timerange.end == start_time + TimeDelta(delta, format='datetime')

    def test_timerange_initialization_with_tuple(self, time_tuple_str):
        start_str, end_str = time_tuple_str
        timerange = TimeRange(time_tuple_str)

        assert timerange.start == Time(start_str)
        assert timerange.end == Time(end_str)

    def test_timerange_equality(self, time_range_example):
        start_time, end_time, delta = time_range_example
        tr1 = TimeRange(start_time, end_time)
        tr2 = TimeRange(start_time, delta)
        assert tr1 == tr2

    def test_timerange_intersection(self, time_range_example):
        start_time, end_time, delta = time_range_example
        tr1 = TimeRange(start_time, delta)
        tr2 = TimeRange(start_time, timedelta(days=2))

        assert tr1.have_intersection(tr2)

    def test_timerange_no_intersection(self, time_range_example):
        start_time, end_time, delta = time_range_example
        tr1 = TimeRange(start_time, end_time)
        tr2 = TimeRange(end_time+delta, timedelta(days=2))

        assert not tr1.have_intersection(tr2)

    def test_timerange_get_dates(self, time_tuple_str):
        tr = TimeRange(time_tuple_str[0], timedelta(days=2))
        dates = tr.get_dates()

        assert len(dates) == 3
        assert dates[0].iso == "2023-01-01 00:00:00.000"
        assert dates[1].iso == "2023-01-02 00:00:00.000"
        assert dates[2].iso == "2023-01-03 00:00:00.000"

    def test_timerange_moving_window(self, time_tuple_str):
        t1_str, t2_str = time_tuple_str
        tr = TimeRange(t1_str, t2_str)
        window_size = TimeDelta(12 * u.hour)
        window_period = TimeDelta(6 * u.hour)

        windows = tr.moving_window(window_size, window_period)

        assert len(windows) == 3
        assert windows[0].start.to_value('iso') == "2023-01-01 00:00:00.000"
        assert windows[0].end.to_value('iso') == "2023-01-01 12:00:00.000"
        assert windows[1].start.to_value('iso') == "2023-01-01 06:00:00.000"
        #TODO time parsing problem
        assert windows[1].end.to_value('iso') == "2023-01-01 18:00:00.000"

    def test_timerange_equal_split(self, time_tuple_str):
        tr = TimeRange(time_tuple_str[0], time_tuple_str[1])
        splits = tr.equal_split(2)

        assert len(splits) == 2
        assert splits[0].start == Time(time_tuple_str[0])
        assert splits[0].end == Time("2023-01-01 12:00:00")
        assert splits[1].start == Time("2023-01-01 12:00:00")
        assert splits[1].end == Time("2023-01-02 00:00:00")

    def test_timerange_shift_forward(self, time_tuple_str):
        tr = TimeRange(time_tuple_str)
        tr.shift_forward(TimeDelta(1 * u.day))
        #todo same problem with iso and isot
        assert tr.start.to_value('iso') == "2023-01-02 00:00:00.000"
        assert tr.end.to_value('iso') == "2023-01-03 00:00:00.000"

    def test_timerange_shift_backward(self):
        tr = TimeRange("2023-01-02 00:00:00", "2023-01-03 00:00:00")
        tr.shift_backward(TimeDelta(1 * u.day))
        # todo same problem with iso and isot
        assert tr.start.to_value('iso') == "2023-01-01 00:00:00.000"
        assert tr.end.to_value('iso') == "2023-01-02 00:00:00.000"

    def test_timerange_extend_range(self, time_tuple_str):
        tr = TimeRange(time_tuple_str)
        tr.extend_range(TimeDelta(-1 * u.day), TimeDelta(1 * u.day))
        # todo same problem with iso and isot
        assert tr.start.to_value('iso') == "2022-12-31 00:00:00.000"
        assert tr.end.to_value('iso') == "2023-01-03 00:00:00.000"