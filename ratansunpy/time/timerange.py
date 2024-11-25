from datetime import timedelta
from datetime import datetime
import astropy.units as u
from typing import List, Optional, Union, Tuple
from astropy.time import Time, TimeDelta
from ratansunpy.time.time import parse_time, check_equal_time

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

__all__ = ['TimeRange']

class TimeRange:
    """
    A class to represent a time range, providing various utilities for time manipulation.

    **Attributes**

    - **start** (`Time`): The start time of the range.
    - **end** (`Time`): The end time of the range.
    - **delta** (`TimeDelta`): The duration of the time range.
    - **days** (`float`): The duration of the time range in days.
    - **hours** (`float`): The duration of the time range in hours.
    - **minutes** (`float`): The duration of the time range in minutes.
    - **seconds** (`float`): The duration of the time range in seconds.

    **Methods**

    - **__init__(a, b=None, format=None)**: Initializes a `TimeRange` object.
    - **__eq__(other)**: Checks if two `TimeRange` objects are equal.
    - **__ne__(other)**: Checks if two `TimeRange` objects are not equal.
    - **__contains__(time)**: Checks if a given time is within the time range.
    - **__repr__()**: Returns a string representation of the `TimeRange` object.
    - **__str__()**: Returns a simple string representation of the start and end times.
    - **have_intersection(other)**: Checks if the time range intersects with another `TimeRange`.
    - **get_dates(filter=None)**: Returns a list of dates within the time range, optionally excluding certain dates.
    - **moving_window(window_size, window_period)**: Returns a list of `TimeRange` objects representing moving windows within the original time range.
    - **equal_split(n_splits=2)**: Splits the time range into equal subranges.
    - **shift_forward(delta=None)**: Shifts the time range forward by a specified delta or its own duration.
    - **shift_backward(delta=None)**: Shifts the time range backward by a specified delta or its own duration.
    - **extend_range(start_delta, end_delta)**: Extends the start and end times of the time range by the specified deltas.
    """
    def __init__(self, a: Union['TimeRange', List[Union[str, Time]], Tuple[Union[str, Time], Union[str, Time]], str, Time],
                 b: Optional[Union[str, Time, timedelta, TimeDelta]] = None,
                 format: Optional[str] = None):
        self._start_time: Optional[Time] = None
        self._end_time: Optional[Time] = None

        # if TimeRange passed 
        if isinstance(a, TimeRange):
            self.__dict__ = a.__dict__.copy()
            return

            # if b is None and a is array-like or tuple.
        # for example, data and delta to the end of timerange
        if b is None:
            x = parse_time(a[0], format=format)
            if len(a) != 2:
                raise ValueError('"a" must have two elements')
            else:
                y = a[1]
        else:
            x = parse_time(a, format=format)
            y = b

            # if y is timedelta
        if isinstance(y, timedelta):
            y = TimeDelta(y, format='datetime')

        # create TimeRange in case of (start_date, delta)
        if isinstance(y, TimeDelta):
            # positive delta 
            if y.jd >= 0:
                self._start_time = x
                self._end_time = x + y
            else:
                self._start_time = x + y
                self._end_time = x
            return

        # otherwise, b is something date-like
        y = parse_time(y, format=format)
        if isinstance(y, Time):
            if x < y:
                self._start_time = x
                self._end_time = y
            else:
                self._start_time = y
                self._end_time = x

    @property
    def start(self):
        return self._start_time

    @property
    def end(self):
        return self._end_time

    @property
    def delta(self):
        return self._end_time - self._start_time

    @property
    def days(self):
        return self.delta.to('day').value

    @property
    def hours(self):
        return self.delta.to('hour').value

    @property
    def minutes(self):
        return self.delta.to('minute').value

    @property
    def seconds(self):
        return self.delta.to('second').value

    def __eq__(self, other: 'TimeRange') -> bool:
        if isinstance(other, TimeRange):
            return check_equal_time(self.start, other.start) and check_equal_time(self.end, other.end)
        return NotImplemented

    def __ne__(self, other: 'TimeRange') -> bool:
        if isinstance(other, TimeRange):
            return not (check_equal_time(self.start, other.start) and check_equal_time(self.end, other.end))
        return NotImplemented

    def __contains__(self, time: Union[str, Time]) -> bool:
        time_to_check = parse_time(time)
        return time_to_check >= self.start and time_to_check <= self.end

    def __repr__(self) -> str:
        start_time = self.start.strftime(TIME_FORMAT)
        end_time = self.end.strftime(TIME_FORMAT)
        full_name = f'{self.__class__.__module__}.{self.__class__.__name__}'
        return (
                f'<{full_name} object at {hex(id(self))}>' +
                '\nStart:'.ljust(12) + start_time +
                '\nEnd:'.ljust(12) + end_time +
                '\nDuration:'.ljust(
                    12) + f'{str(self.days)} days | {str(self.hours)} hours | {str(self.minutes)} minutes | {str(self.seconds)} seconds'
        )

    def __str__(self) -> str:
        start_time = self.start.strftime(TIME_FORMAT)
        end_time = self.end.strftime(TIME_FORMAT)
        return (
            f'({start_time}, {end_time})'
        )

    def have_intersection(self, other: 'TimeRange') -> bool:
        """
        Checks if the time range intersects with another TimeRange.

       :param other: Another TimeRange object to compare.
       :type other: TimeRange
       :returns: True if the ranges intersect, False otherwise.
       :rtype: bool``

        """

        intersection_lower = max(self.start, other.start)
        intersection_upper = min(self.end, other.end)
        return intersection_lower <= intersection_upper

    def get_dates(self, filter: Optional[List[datetime]] = None) -> List[datetime]:
        """
        Generate a list of dates within the time range defined by the instance.

        :param filter: A list of dates to be excluded from the result. If provided, only dates not in this list will be included.
        :type filter: list of datetime-like objects or None
        :return: A list of dates from start to end of the time range, with optional exclusion.
        :rtype: list of datetime-like objects
        """
        delta = self.end.to_datetime().date() - self.start.to_datetime().date()
        t_format = "%Y-%m-%d"
        dates_list = [parse_time(self.start.strftime(t_format)) + TimeDelta(i * u.day)
                      for i in range(delta.days + 1)]
        # filter is a list of dates to be excluded, maybe should add typings
        if filter:
            dates_list = [date for date in dates_list if date not in parse_time(filter)]
        return dates_list

    def moving_window(self, window_size: Union[TimeDelta, int], window_period: Union[TimeDelta, int]) -> List['TimeRange']:
        """
        Generate a list of time ranges using a moving window approach.

        :param window_size: The duration of each time window.
        :type window_size: TimeDelta or int
        :param window_period: The period to shift the window after each iteration.
        :type window_period: TimeDelta or int
        :return: A list of `TimeRange` objects representing the moving windows.
        :rtype: list of TimeRange
        """
        if not isinstance(window_size, TimeDelta):
            window_size = TimeDelta(window_size)
        if not isinstance(window_period, TimeDelta):
            window_period = TimeDelta(window_period)

        window_number = 1
        times = [TimeRange(self.start, self.start + window_size)]

        while times[-1].end < self.end:
            times.append(
                TimeRange(
                    self.start + window_number * window_period,
                    self.start + window_number * window_period + window_size,
                )
            )
            print(times)
            window_number += 1
        return times

    def equal_split(self, n_splits: int = 2) -> List['TimeRange']:
        """
        Split the time range into equal subranges.

        :param n_splits: The number of subranges to divide the time range into. Must be greater than or equal to 1.
        :type n_splits: int
        :raises ValueError: If `n_splits` is less than or equal to 0.
        :return: A list of `TimeRange` objects representing the equal subranges.
        :rtype: list of TimeRange
        """
        if n_splits <= 0:
            raise ValueError('n must be greater or equal than 1')
        subranges = []
        prev_time = self.start
        next_time = None
        for _ in range(n_splits):
            next_time = prev_time + self.delta / n_splits
            next_range = TimeRange(prev_time, next_time)
            subranges.append(next_range)
            prev_time = next_time
        return subranges

    def shift_forward(self, delta: Optional[TimeDelta] = None) -> 'self':
        """
        Shift the entire time range forward by a specified duration.

        :param delta: The duration by which to shift the time range. If not provided, the entire duration of the time range is used.
        :type delta: TimeDelta or None
        :return: The instance with the updated time range.
        :rtype: self
        """
        delta = delta if delta else self.delta
        self._start_time += delta
        self._end_time += delta
        return self

    def shift_backward(self, delta: Optional[TimeDelta] = None) -> 'self':
        """
        Shift the entire time range backward by a specified duration.

        :param delta: The duration by which to shift the time range. If not provided, the entire duration of the time range is used.
        :type delta: TimeDelta or None
        :return: The instance with the updated time range.
        :rtype: self
        """
        delta = delta if delta else self.delta
        self._start_time -= delta
        self._end_time -= delta
        return self

    def extend_range(self, start_delta: TimeDelta, end_delta: TimeDelta) -> 'self':
        """
        Extend the time range by modifying the start and end times.

        :param start_delta: The amount of time to add to the start of the range.
        :type start_delta: TimeDelta
        :param end_delta: The amount of time to add to the end of the range.
        :type end_delta: TimeDelta
        :return: The instance with the extended time range.
        :rtype: self
        """
        self._start_time += start_delta
        self._end_time += end_delta
        return self
