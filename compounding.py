import pandas as pd
import numpy as np
from datetime import datetime
from pandas.core.generic import NDFrame
from cardano.pystats.constants import TimePeriods as tp
from dateutil.relativedelta import relativedelta


class CompoundingTime:

    def __init__(self, optional_dates: dict = None):
        """
        helper class containing date time info which is
        passed to the compounding routine
        :param optional_dates: a dictionary of dates to calculate
        period returns from
        """

        self.optional_dates = optional_dates if optional_dates is not None else {}

    def get_compounding_dates(self,
                              selected_date: datetime,
                              start_date: datetime):

        """
        Returns the relevant durations and their starting dates given the currently selected date
        and the start date of the return data - here, a relevant duration is one in which the whole
        of the period is after the start date.  The method implicitly assumes that all dates save for
        the latest are month end

        Input
        ------
        selected_date -> datetime object which shows the selected end date (ie the date to which all
        of the durations are measured)
        start_date -> datetime object which shows the start date of the returns data

        Output
        ------
        returns a dictionary which contains key value pairs of time periods and their start date (as a timestamp),
        where any durations which would extend beyond the start date are omitted

        """

        output = dict()

        if selected_date < start_date:
            return output

        qtd_start = min(selected_date + pd.offsets.QuarterEnd(-1) + pd.offsets.MonthEnd(1), selected_date)
        ytd_start = min(selected_date + pd.offsets.YearEnd(-1) + pd.offsets.MonthEnd(1), selected_date)
        three_mo_start = selected_date + pd.offsets.MonthEnd(-2)
        twelve_mo_start = selected_date + pd.offsets.MonthEnd(-11)

        output.update({tp.MONTH_TO_DATE: pd.Timestamp(selected_date)})
        if qtd_start >= start_date:
            output.update({tp.QUARTER_TO_DATE: qtd_start})
        if ytd_start >= start_date:
            output.update({tp.YEAR_TO_DATE: ytd_start})
        if three_mo_start >= start_date:
            output.update({tp.THREE_MONTHS: three_mo_start})
        if twelve_mo_start >= start_date:
            output.update({tp.TWELVE_MONTHS: twelve_mo_start})

        for date_name in self.optional_dates:

            date = self.optional_dates[date_name]

            if selected_date >= date >= start_date:
                output.update({date_name: date})

        return output

    @staticmethod
    def get_fixed_annualisations():
        """
        returns a dict of periods and the months
        required for annualising
        """

        # currently none of these, might be at a later point
        # eg when we have 3 yr history

        return {}


def remove_undefined_contributions(compounded_returns: pd.DataFrame, perf_name: str):
    """
    utiltiy method for stripping out the zero returns (instrument not held at
    a particular date) and null returns (time period too short)
    :param compounded_returns: a pandas dataframe of the compounded returns
    :param perf_name: name of the column containing the performance data
    :return:
    """

    mask = (compounded_returns[perf_name] != 0) & pd.notnull(compounded_returns[perf_name])
    compounded_returns = compounded_returns[mask]
    return compounded_returns


def get_total_compound_conts_to_pfo(returns_data: pd.DataFrame,
                                    optional_dates: dict = None,
                                    portfolio_returns: pd.Series = None):
    """
    wrapper function for main compounding routine where we DON'T want to annualise the returns
    """

    return _get_compound_contributions(returns_data, False, optional_dates, portfolio_returns)


def get_compound_conts_to_pfo_annualised(returns_data: pd.DataFrame,
                                         optional_dates: dict = None,
                                         portfolio_returns: pd.Series = None):
    """
    wrapper function for main compounding routine where we DO want to annualise the returns
    """
    return _get_compound_contributions(returns_data, True, optional_dates, portfolio_returns)


def _get_compound_contributions(returns_data: pd.DataFrame,
                                annualise_returns: bool,
                                optional_dates: dict = None,
                                portfolio_returns: pd.Series = None):
    """"

    Calculates the compounded contribution of an asset (or assets) to overall portfolio return over
    the following time periods: MTD, QTD, YTD, 3 month, 12 month and since SCA inception.  This is done by
    constructing a matrix of cumulative whole-portfolio returns to the end of the period (base_adj_df) then adjusting
    this matrix(to get adj_df) so that only the relevant elements for each different time period are non-zero.
    Multiplying the returns data by adj_df then gives the required contributions over each time period to the
    selected date (in dataframe form).

    Input
    ------
    returns_data -> this should be a pandas dataframe comprised of timeseries, where the index is the dates and
                    the columns the different theses.  The function is designed to take monthly data - all
                    dates save for the final date must be month end values.  The series should be % contribution
                    series, not £ contribution series.

    portfolio_returns -> a single time series which represents the total return on the portfolio.  The dates should
                         the same as those in the returns_data, and again this should be % contributions not
                         £ contributions. If this is None, it will be calculated by summing the returns for each period
                         in the returns_data.

    Output
    ------
    the output will be a dataframe with the following columns: Thesis, Date, and one column for each duration
    """

    if portfolio_returns is None:
        portfolio_returns = returns_data.sum(axis=1)
    returns_data = returns_data.fillna(value=0)
    selected_date, start_date = returns_data.index.max(), returns_data.index.min()
    ct = CompoundingTime(optional_dates)
    dates = ct.get_compounding_dates(selected_date, start_date)
    base_adj_df = pd.concat([portfolio_returns] * len(dates), axis=1) + 1
    base_adj_df.columns = dates.keys()
    base_adj_df = _get_cum_prod_to_last_date(base_adj_df)
    compounded_returns, date_list = [], []
    n_items = returns_data.shape[1]

    while True:

        adj_df = _adjust_cum_prod_by_duration(base_adj_df.copy(), dates)
        contributions = returns_data.T.dot(adj_df)
        date_list = date_list + [selected_date] * n_items
        compounded_returns.append(contributions)

        if start_date == selected_date:
            break

        selected_date = returns_data.index[-2]
        dates = ct.get_compounding_dates(selected_date, start_date)
        base_adj_df = _rollback_cum_prod(base_adj_df, dates)
        returns_data = returns_data[:-1]

    output = pd.concat(compounded_returns)
    output[tp.DATE_NAME] = date_list

    if annualise_returns:
        output = _get_annualised_conts(output, ct)

    return output


def _adjust_cum_prod_by_duration(base_adj_df: pd.DataFrame, dates: dict):
    """
    returns the matrix of top-level performance figures necessary to compound the
    individual thesis returns. The array should contain returns data

    Inputs
    ------
    base_adj_df -> dataframe containing the whole portfolio returns time series to the selected
    dated, repeated once for each relevant time period
    dates -> a dictionary containing the relevant durations and the corresponding start dates

    Outputs
    ------
    returns a dataframe representing the adjustment matrix required to compound the returns.  Each column corresponds
    to a particular duration, and will have entries which are either zero (for dates prior to the start date) or instead
    represent the return achieved by the portfolio over the remaining period from the relevant date
    """

    for col in base_adj_df.columns:
        date = dates[col]
        base_adj_df[col].where(base_adj_df.index.to_series() >= date, other=0, inplace=True)

    return base_adj_df


def _rollback_cum_prod(base_adj_df: pd.DataFrame, dates: dict):
    """
    takes in the base adjustment dataframe and "rolls it back" for the next time period, removing the
    last row and then rescaling such that the new final row is all 1s

    Inputs
    ------
    base_adj_df -> dataframe containing the whole portfolio returns time series to the selected
    dated, repeated once for each relevant time period
    dates -> a dictionary containing the relevant durations and the corresponding start dates

    Outputs
    ------
    returns a new base adjustment matrix, with the bottom column removed and all entries rescaled
    """

    base_adj_df = base_adj_df[list(dates.keys())]
    base_adj_df = base_adj_df.drop(base_adj_df.index[-1])
    base_adj_df = base_adj_df.multiply(1 / base_adj_df.iloc[-1, 0])

    return base_adj_df


def _get_cum_prod_to_last_date(df: pd.DataFrame):
    """
    helper function which generates an array where each element is the product of all those
    below it (in its column).  Necessary for computing the adjustment matrix in the compounding
    routine. If the array contains any zero elements, an error is thrown

    Inputs
    ------
    df -> a pandas dataframe

    Outputs
    ------
    returns a numpy array of the same dimensions, calculated as above
    """

    if (df.values == 0).any():
        raise ValueError('inverse_cum__prod requires a totally non-zero array')

    prod = df.values.prod(axis=0)
    prod_array = np.stack([prod] * df.shape[0])
    cum_prod_df = df.cumprod(axis=0)

    return prod_array / cum_prod_df


def _annualise_returns(returns: NDFrame, months_for_annualise: int):
    """
    annualise a column of returns based on the return input
    and months to annualise
    :param returns: a series of returns, each of which requires
    annualisng
    :param months_for_annualise: a series of numbers of months to use in the
    annualisation, or a single scalar value
    """

    return (1 + returns) ** (12 / months_for_annualise) - 1


def _get_annualised_conts(conts: pd.DataFrame,
                          ct: CompoundingTime):
    """
    annualise contributions of periods greater
    than one year
    :param conts: contribution output from the main compounding
    routine
    :param ct: a CompoudingTime object
    """

    # annualise fixed periods

    fixed_annualisations = ct.get_fixed_annualisations()

    for period in fixed_annualisations:
        months = fixed_annualisations[period]
        conts[period] = _annualise_returns(conts[period], months)

    # annualise optional dates periods

    for period in ct.optional_dates:
        date = ct.optional_dates[period]

        def months_for_annualise(dt):
            months_ann = relativedelta(dt, date).months + 1 + \
                relativedelta(dt, date).years * 12
            return max(months_ann, 12)

        months = conts[tp.DATE_NAME].apply(months_for_annualise)
        conts[period] = _annualise_returns(conts[period], months)

    return conts
