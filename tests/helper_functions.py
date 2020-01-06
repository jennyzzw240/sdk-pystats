import pandas as pd
from datetime import datetime


def is_third_friday_of_month(date: datetime):
    """
    Returns true if the input datetime is the third friday of the month,
    and returns false otherwise.

    Input(s)
    --------
    date -> a datetime

    Output(s)
    --------
    returns true if the date is the third friday of the month, else false
    """

    weekday = date.weekday()
    day = date.day
    minimum_day = 14
    maximum_day = 22
    # Is the weekday a Friday and is the day within the third week
    return weekday == 4 and minimum_day < day < maximum_day


def third_friday_of_month_dates(start_date: datetime,
                                end_date: datetime):
    """
    Returns a list containing all dates between the start and end
    input (inclusive) which represent the third friday of a month

    Input(s)
    --------
    start_date, end_date -> datetime objects denoting the start
    and end of the search interval (inclusive)

    Output(s)
    --------
    returns a list containing the dates which are the third friday of a
    month as pandas timestamp objects
    """

    date_range = pd.bdate_range(start_date, end_date, freq='b')
    third_fridays = [x for x in date_range if is_third_friday_of_month(x)]
    return third_fridays
