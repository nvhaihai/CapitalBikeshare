import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from typing import List
from pandas.tseries import offsets


class TimeFeature:
    def __call__(self, dates):
        raise NotImplementedError


class MonthOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.month


class DayOfMonth(TimeFeature):
    def __call__(self, dates):
        return dates.day


class WeekOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.isocalendar().week


class DayOfWeek(TimeFeature):
    def __call__(self, dates):
        return dates.dayofweek


class DayOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.dayofyear


class HourOfDay(TimeFeature):
    def __call__(self, dates):
        return dates.hour


class MinuteOfHour(TimeFeature):
    def __call__(self, dates):
        return dates.minute


class SecondOfMinute(TimeFeature):
    def __call__(self, dates):
        return dates.second


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from typing import List
from pandas.tseries import offsets


class TimeFeature:
    def __call__(self, dates):
        raise NotImplementedError


class MonthOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.month


class DayOfMonth(TimeFeature):
    def __call__(self, dates):
        return dates.day


class WeekOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.isocalendar().week


class DayOfWeek(TimeFeature):
    def __call__(self, dates):
        return dates.dayofweek


class DayOfYear(TimeFeature):
    def __call__(self, dates):
        return dates.dayofyear


class HourOfDay(TimeFeature):
    def __call__(self, dates):
        return dates.hour


class MinuteOfHour(TimeFeature):
    def __call__(self, dates):
        return dates.minute


class SecondOfMinute(TimeFeature):
    def __call__(self, dates):
        return dates.second


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
