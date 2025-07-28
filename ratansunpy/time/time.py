import re
from typing import Optional, Union, List, Any
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import singledispatch
from dataclasses import dataclass
import numpy as np
import pandas as pd
import astropy.time
import astropy.units as u
from astropy.time import Time

TIME_REGEX = {
    '%Y': r'(?P<year>\d{4})',
    '%j': r'(?P<dayofyear>\d{3})',
    '%m': r'(?P<month>\d{1,2})',
    '%d': r'(?P<day>\d{1,2})',
    '%H': r'(?P<hour>\d{1,2})',
    '%M': r'(?P<minute>\d{1,2})',
    '%S': r'(?P<second>\d{1,2})',
    '%f': r'(?P<microsecond>\d+)',
    '%b': r'(?P<month_str>[a-zA-Z]+)',
}


@dataclass(frozen=True)
class TimeFormats:
    common_formats: List[str] = field(default_factory=lambda: (
        "%Y-%m-%dT%H:%M:%S",  # 2007-05-04T21:08:12
        "%Y/%m/%dT%H:%M:%S",  # 2007/05/04T21:08:12
        "%Y%m%dT%H%M%S",  # 20070504T210812
        "%Y/%m/%d %H:%M:%S",  # 2007/05/04 21:08:12
        "%Y/%m/%d %H:%M",  # 2007/05/04 21:08
        "%Y-%m-%d %H:%M:%S",  # 2007-05-04 21:08:12
        "%Y-%m-%d %H:%M",  # 2007-05-04 21:08
        "%Y-%b-%d %H:%M:%S",  # 2007-May-04 21:08:12
        "%Y-%b-%d %H:%M",  # 2007-May-04 21:08
        "%Y-%b-%d",  # 2007-May-04
        "%Y-%m-%d",  # 2007-05-04
        "%Y/%m/%d",  # 2007/05/04
        "%d-%b-%Y",  # 04-May-2007
        "%d-%b-%Y %H:%M:%S",  # 04-May-2007 21:08:12
        "%Y%m%d_%H%M%S",  # 20070504_210812
        "%Y:%j:%H:%M:%S",  # 2012:124:21:08:12
        "%Y%m%d%H%M%S",  # 20140101000001 (JSOC/VSO Export/Downloads)
        "%Y.%m.%d_%H:%M:%S_TAI",  # 2016.05.04_21:08:12_TAI - JSOC
        "%Y.%m.%d_%H:%M:%S_UTC",  # 2016.05.04_21:08:12_UTC - JSOC
        "%Y.%m.%d_%H:%M:%S",  # 2016.05.04_21:08:12 - JSOC
        "%Y/%m/%dT%H:%M",  # 2007/05/04T21:08
        "%Y-%m-%dT%H:%M:%S.%f",  # 2007-05-04T21:08:12.999999
        "%Y/%m/%dT%H:%M:%S.%f",  # 2007/05/04T21:08:12.999999
        "%Y-%m-%dT%H:%M:%S.%fZ",  # 2007-05-04T21:08:12.999Z
        "%Y%m%dT%H%M%S.%f",  # 20070504T210812.999999
         "%Y/%m/%d %H:%M:%S.%f",  # 2007/05/04 21:08:12.999999
        "%Y-%m-%d %H:%M:%S.%f",  # 2007-05-04 21:08:12.999999
        "%Y-%b-%d %H:%M:%S.%f",  # 2007-May-04 21:08:12.999999
        "%d-%b-%Y %H:%M:%S.%f",  # 04-May-2007 21:08:12.999999
        "%Y:%j:%H:%M:%S.%f",  # 2012:124:21:08:12.999999
    ))
    examples: List[str] = field(default_factory=lambda: (
        "2007-05-04T21:08:12",
        "2007/05/04T21:08:12",
        "20070504T210812",
        "2007/05/04 21:08:12",
        "2007/05/04 21:08",
        "2007-05-04 21:08:12",
        "2007-05-04 21:08",
        "2007-May-04 21:08:12",
        "2007-May-04 21:08",
        "2007-May-04",
        "2007-05-04",
        "2007/05/04",
        "04-May-2007",
        "04-May-2007 21:08:12",
        "20070504_210812",
        "2012:124:21:08:12",
        "20140101000001", #(JSOC/VSO Export/Downloads)
        "2016.05.04_21:08:12_TAI",  #JSOC
        "2016.05.04_21:08:12_UTC",  #JSOC
        "2016.05.04_21:08:12",  #JSOC
        "2007/05/04T21:08",
        "2007-05-04T21:08:12.999999",
        "2007/05/04T21:08:12.999999",
        "2007-05-04T21:08:12.999Z",
        "20070504T210812.999999",
        "2007/05/04 21:08:12.999999",
        "2007-05-04 21:08:12.999999",
        "2007-May-04 21:08:12.999999",
        "04-May-2007 21:08:12.999999",
        "2012:124:21:08:12.999999",
    ))


COMMON_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",  # 2007-05-04T21:08:12
    "%Y/%m/%dT%H:%M:%S",  # 2007/05/04T21:08:12
    "%Y%m%dT%H%M%S",  # 20070504T210812
    "%Y/%m/%d %H:%M:%S",  # 2007/05/04 21:08:12
    "%Y/%m/%d %H:%M",  # 2007/05/04 21:08
    "%Y-%m-%d %H:%M:%S",  # 2007-05-04 21:08:12
    "%Y-%m-%d %H:%M",  # 2007-05-04 21:08
    "%Y-%b-%d %H:%M:%S",  # 2007-May-04 21:08:12
    "%Y-%b-%d %H:%M",  # 2007-May-04 21:08
    "%Y-%b-%d",  # 2007-May-04
    "%Y-%m-%d",  # 2007-05-04
    "%Y/%m/%d",  # 2007/05/04
    "%d-%b-%Y",  # 04-May-2007
    "%d-%b-%Y %H:%M:%S",  # 04-May-2007 21:08:12
    "%Y%m%d_%H%M%S",  # 20070504_210812
    "%Y:%j:%H:%M:%S",  # 2012:124:21:08:12
    "%Y%m%d%H%M%S",  # 20140101000001 (JSOC/VSO Export/Downloads)
    "%Y.%m.%d_%H:%M:%S_TAI",  # 2016.05.04_21:08:12_TAI - JSOC
    "%Y.%m.%d_%H:%M:%S_UTC",  # 2016.05.04_21:08:12_UTC - JSOC
    "%Y.%m.%d_%H:%M:%S",  # 2016.05.04_21:08:12 - JSOC
    "%Y/%m/%dT%H:%M",  # 2007/05/04T21:08
    "%Y-%m-%dT%H:%M:%S.%f",  # 2007-05-04T21:08:12.999999
    "%Y/%m/%dT%H:%M:%S.%f",  # 2007/05/04T21:08:12.999999
    "%Y-%m-%dT%H:%M:%S.%fZ",  # 2007-05-04T21:08:12.999Z
    "%Y%m%dT%H%M%S.%f",  # 20070504T210812.999999
     "%Y/%m/%d %H:%M:%S.%f",  # 2007/05/04 21:08:12.999999
    "%Y-%m-%d %H:%M:%S.%f",  # 2007-05-04 21:08:12.999999
    "%Y-%b-%d %H:%M:%S.%f",  # 2007-May-04 21:08:12.999999
    "%d-%b-%Y %H:%M:%S.%f",  # 04-May-2007 21:08:12.999999
    "%Y:%j:%H:%M:%S.%f",  # 2012:124:21:08:12.999999
]


def check_equal_time(t1: Time, t2: Time) -> bool:
    """
    Check if two time objects are equal within a nanosecond precision.

    :param t1: The first time object.
    :type t1: Time
    :param t2: The second time object.
    :type t2: Time
    :returns: True if the time difference is less than a nanosecond, False otherwise.
    :rtype: bool

    **Example:**

    .. code-block:: python

        >>> from astropy.time import Time
        >>> t1 = Time('2020-01-01T00:00:00.000000000')
        >>> t2 = Time('2020-01-01T00:00:00.000000001')
        >>> check_equal_time(t1, t2)  # Returns: True
    """
    if abs(t2 - t1) < 1 * u.nanosecond:
        return True
    return False


def regex_time(time_string: str, format: str) -> Optional[str]:
    """
    Match a time string against a format using regular expressions.

    :param time_string: The time string to match.
    :type time_string: str
    :param format: The time format string.
    :type format: str
    :returns: The time string if a match is found, otherwise None.
    :rtype: Optional[str]

    **Example:**

    .. code-block:: python

       >>>  regex_time('2020-01-01T00:00:00.000000', '%Y-%m-%dT%H:%M:%S.%f')
        # Returns: '2020-01-01T00:00:00.000000'
    """
    regex_pattern = format
    for key, value in TIME_REGEX.items():
        regex_pattern = regex_pattern.replace(key, value)
    # Обработка необязательного микросекундного компонента
    if '%f' in format:
        regex_pattern = regex_pattern.replace(r'(?P<microsecond>\d{1,6})', r'(?P<microsecond>\d{1,6})?')
    regex_pattern = '^' + regex_pattern + '$'  # Добавим начало и конец строки
    match = re.match(regex_pattern, time_string)
    if match is None:
        return None
    return time_string


def get_time_format(time_string: str) -> Optional[str]:
    """
    Determine the time format of a given time string.

    :param time_string: The time string to check.
    :type time_string: str
    :returns: The format string if a matching format is found, otherwise None.
    :rtype: Optional[str]

    :Example:

    .. code-block:: python

        >>> get_time_format('2020-01-01T00:00:00.000000')
        Returns: '%Y-%m-%dT%H:%M:%S.%f'
    """
    for time_format in COMMON_TIME_FORMATS:
        if regex_time(time_string, time_format) is not None:
            return time_format
    return None


@singledispatch
def create_time(time_string: Any, format: Optional[str] = None, **kwargs) -> Time:
    """
    Create an Astropy Time object from various time representations.

    :param time_string: The time data, which can be of various types.
    :type time_string: Any
    :param format: The time format string, if known.
    :type format: Optional[str]
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> create_time('2020-01-01T00:00:00.000000')
        # Returns: <Time object: scale='utc' format='isot' value=2020-01-01T00:00:00.000>
    """
    return Time(time_string, format=format, **kwargs)


@create_time.register(pd.Timestamp)
def create_time_pandasTimestamp(time_string: pd.Timestamp, **kwargs) -> Time:
    """
    Convert a pandas Timestamp to an Astropy Time object.

    :param time_string: A pandas Timestamp object.
    :type time_string: pd.Timestamp
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> ts = pd.Timestamp('2020-01-01T00:00:00')
        >>> create_time(ts)
        <Time object: scale='utc' format='datetime64' value=2020-01-01T00:00:00.000>
    """
    return Time(time_string.asm8)


@create_time.register(pd.Series)
def create_time_pandasSeries(time_string: pd.Series, **kwargs) -> Time:
    """
    Convert a pandas Series of timestamps to an Astropy Time object.

    :param time_string: A pandas Series of timestamp objects.
    :type time_string: pd.Series
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> ts_series = pd.Series([pd.Timestamp('2020-01-01T00:00:00')])
        >>> create_time(ts_series)
        <Time object: scale='utc' format='datetime64' value=[2020-01-01T00:00:00.000]>
    """
    return Time(time_string.tolist(), **kwargs)


@create_time.register(pd.DatetimeIndex)
def create_time_pandasDatetimeIndex(time_string: pd.DatetimeIndex, **kwargs) -> Time:
    """
    Convert a pandas DatetimeIndex to an Astropy Time object.

    :param time_string: A pandas DatetimeIndex object.
    :type time_string: pd.DatetimeIndex
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> dt_index = pd.DatetimeIndex(['2020-01-01T00:00:00'])
        >>> create_time(dt_index)
        <Time object: scale='utc' format='datetime64' value=[2020-01-01T00:00:00.000]>
    """
    return Time(time_string.tolist(), **kwargs)


@create_time.register(datetime)
def create_time_datetime(time_string: datetime, **kwargs) -> Time:
    """
    Convert a datetime object to an Astropy Time object.

    :param time_string: A datetime object.
    :type time_string: datetime
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> dt = datetime(2020, 1, 1)
        >>> create_time(dt)
        <Time object: scale='utc' format='datetime' value=2020-01-01T00:00:00.000>
    """
    return Time(time_string, **kwargs)


@create_time.register(date)
def create_time_date(time_string: date, **kwargs) -> Time:
    """
    Convert a date object to an Astropy Time object.

    :param time_string: A date object.
    :type time_string: date
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> d = date(2020, 1, 1)
        >>> create_time(d)
        <Time object: scale='utc' format='iso' value=2020-01-01T00:00:00.000>
    """
    return Time(time_string.isoformat(), **kwargs)


@create_time.register(np.datetime64)
def create_time_npdatetime64(time_string: np.datetime64, **kwargs) -> Time:
    """
    Convert a numpy datetime64 object to an Astropy Time object.

    :param time_string: A numpy datetime64 object.
    :type time_string: np.datetime64
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> dt64 = np.datetime64('2020-01-01T00:00:00')
        >>> create_time(dt64)
        <Time object: scale='utc' format='iso' value=2020-01-01T00:00:00.000>
    """
    return Time(str(time_string.astype('M8[ns]')), **kwargs)


@create_time.register(np.ndarray)
def create_time_npndarray(time_string: np.ndarray, **kwargs) -> Time:
    """
    Convert a numpy ndarray of datetime64 objects to an Astropy Time object.

    :param time_string: A numpy ndarray of datetime64 objects.
    :type time_string: np.ndarray
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> dt64_array = np.array(['2020-01-01T00:00:00'], dtype='datetime64')
        >>> create_time(dt64_array)
        <Time object: scale='utc' format='iso' value=[2020-01-01T00:00:00.000]>
    """
    if 'datetime64' in str(time_string.dtype):
        return Time([str(dt.astype('M8[ns]')) for dt in time_string], **kwargs)
    else:
        return create_time.dispatch(object)(time_string, **kwargs)


@create_time.register(astropy.time.Time)
def create_time_astropy(time_string: Time, **kwargs) -> Time:
    """
    Return the provided Astropy Time object.

    :param time_string: An Astropy Time object.
    :type time_string: Time
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: The provided Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> t = Time('2020-01-01T00:00:00')
        >>> create_time(t)
        <Time object: scale='utc' format='isot' value=2020-01-01T00:00:00.000>
    """
    return time_string


@create_time.register(list)
def create_time_list(time_list: list, format: Optional[str] = None, **kwargs) -> Time:
    """
    Convert a list of time strings to an Astropy Time object.

    :param time_list: A list of time strings.
    :type time_list: list
    :param format: The time format string, if known.
    :type format: Optional[str]
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> time_list = ['2020-01-01T00:00:00']
        >>> create_time(time_list)
        <Time object: scale='utc' format='isot' value=[2020-01-01T00:00:00.000]>
    """
    item = time_list[0]
    if isinstance(item, str) and format is None:
        time_format = get_time_format(item)
        return Time.strptime(time_list, time_format, **kwargs)

    return create_time_list.dispatch(object)(time_list, format, **kwargs)


@create_time.register(str)
def create_time_str(time_string: str, **kwargs) -> Time:
    """
    Convert a time string to an Astropy Time object using common time formats.

    :param time_string: A time string.
    :type time_string: str
    :param kwargs: Additional keyword arguments passed to ``astropy.time.Time``.
    :returns: An Astropy Time object.
    :rtype: Time

    **Example:**

    .. code-block:: python

        >>> create_time('2020-01-01T00:00:00')
        <Time object: scale='utc' format='isot' value=2020-01-01T00:00:00.000>
    """
    for time_format in COMMON_TIME_FORMATS:
        try:
            try:
                ts = regex_time(time_string, time_format)
            except TypeError:
                break
            if ts is None:
                continue
            t = Time.strptime(ts, time_format, **kwargs)
            return t
        except ValueError:
            pass

    return create_time.dispatch(object)(time_string, **kwargs)


def parse_time(time_string, format=None, **kwargs):
    if isinstance(time_string, str) and time_string == 'now':
        rt = Time.now()
    else:
        rt = create_time(time_string, format=format, **kwargs)
    return rt
