import numpy as np
from pandas import Series
from pandas.core.generic import NDFrame
from cardano.pystats.constants import TimePeriods as tp


class TSeriesHelper:

    @staticmethod
    def price_to_return(price_series: NDFrame):
        """
        convert a series of asset prices to a series
        or returns
        :param price_series: pandas price series or dataframe
        of price series
        :return:
        """

        if (price_series == 0).any(axis=None).any():
            raise ValueError("Cannot convert price series with zeroes to a return")

        price_series = price_series / price_series.shift(1)
        price_series.dropna(inplace=True)
        price_series = price_series - 1

        return price_series

    @staticmethod
    def return_to_price(return_series: NDFrame, init_value: int = 100):
        """
        convert a series of asset prices to a series
        or returns
        :param return_series: pandas return series or dataframe
        of return series
        :param init_value: the initial price of the price
        series to be returned. Default is 100
        :return:
        """

        cum_returns = (1 + return_series).cumprod()
        price_series = cum_returns * init_value
        return price_series

    @staticmethod
    def get_sortino_ratio_unscaled(price_series: NDFrame, benchmark_series: Series):
        """"
        Returns the Sortino ratio, with no scaling applied
        to annualise the statistic - hence must be comapring
        price series of the same frequency

        :param price_series - a pandas series or data frame of prices
        :param benchmark_series - a pandas series representing the total
        return/price series of the chosen benchmark
        """

        return TSeriesHelper._get_sortino_ratio(price_series, benchmark_series, False)

    @staticmethod
    def get_sortino_ratio_scaled(price_series: NDFrame, benchmark_series: Series):
        """"
        Returns the Sortino ratio, scaled by sqrt(time) to

        :param price_series - a pandas series or data frame of prices
        :param benchmark_series - a pandas series representing the total
        return/price series of the chosen benchmark
        """

        return TSeriesHelper._get_sortino_ratio(price_series, benchmark_series, True)

    @staticmethod
    def get_sharpe_ratio_unscaled(price_series: NDFrame, rf_series: Series):
        """
        get the sharpe ratio with no scaling applied
        to annualise it - hence must be comparing price
        series of the same frequency
        :param price_series - a pandas series or data frame of prices
        :param rf_series - a pandas series representing the total
        return/price series of the risk free rate
        :return:
        """

        return TSeriesHelper._get_sharpe_ratio(price_series, rf_series, False)

    @staticmethod
    def get_sharpe_ratio_scaled(price_series: NDFrame, rf_series: Series):
        """
        get the sharpe ratio scaled by sqrt(time) to annualise
        the statistic
        :param price_series - a pandas series or data frame of prices
        :param rf_series - a pandas series representing the total
        return/price series of the risk free rate
        :return:
        """

        return TSeriesHelper._get_sharpe_ratio(price_series, rf_series, True)

    @staticmethod
    def get_annualised_vol(price_series: NDFrame):
        """"
        Returns the annualised volatility of returns based on a stream of asset prices.  The function derives
        the time period of the price data uses this to calculate a suitable factor to annualise the data with.
        Note that the function calculates the standard deviation of the NATURAL LOG of the returns, as is conventional.
        :param price_series - a pandas series or dataframe of prices

        Notes
        --------
        https://en.wikipedia.org/wiki/Volatility_(finance)
        """

        log_returns = np.log(price_series / price_series.shift(1))
        annualising_scaling = TSeriesHelper._get_annualisation_factor(price_series.index)
        return np.std(log_returns) * annualising_scaling

    @staticmethod
    def get_annualised_returns(price_series: NDFrame):
        """"
        Calculates the return based on a stream of asset prices - the result is annualised for
        periods over 1 year
        :param price_series - a pandas series or dataframe of prices
        """

        no_of_returns = len(price_series)
        total_return = (price_series.iloc[no_of_returns - 1] / price_series.iloc[0])

        years = (price_series.index.max() - price_series.index.min()).days / tp.D_PER_ANUM

        exponent = min(1 / years, 1)
        annualised_return = total_return ** exponent - 1

        return annualised_return

    @staticmethod
    def get_annualised_excess_return(price_series: NDFrame, rf_series: Series):
        """
        calculates the excess return of the assets over the risk
        free rate, annualised for periods of greater than one
        year

        :param price_series: a pandas series or data frame of prices
        over time
        :param rf_series: annualised risk free return over the period
        under consideration
        :return:
        """
        annualised_strategy_return = TSeriesHelper.get_annualised_returns(price_series)
        annualised_rf_return = TSeriesHelper.get_annualised_returns(rf_series)
        excess_return = (annualised_strategy_return - annualised_rf_return)
        return excess_return

    @staticmethod
    def _get_sortino_ratio(price_series: NDFrame,
                           benchmark_series: Series,
                           scale_to_annualise: bool):
        """"
        Returns the Sortino ratio

        :param price_series - a pandas series or data frame of prices
        :param benchmark_series - a pandas series representing the total
        return/price series of the chosen benchmark
        Notes
        ------
        https://en.wikipedia.org/wiki/Sortino_ratio
        """
        returns = (price_series / price_series.shift(1)).dropna()
        returns_rf = (benchmark_series / benchmark_series.shift(1)).dropna()

        avg_excess = (returns.subtract(returns_rf, axis=0)).mean()
        vol = TSeriesHelper._get_downside_deviation(returns, returns_rf)
        annualising_scaling = TSeriesHelper._get_annualisation_factor(price_series.index) if scale_to_annualise else 1

        sortino_ratio = avg_excess * annualising_scaling / vol

        return sortino_ratio

    @staticmethod
    def _get_sharpe_ratio(price_series: NDFrame,
                          rf_series: Series,
                          scale_to_annualise: bool):
        """"
        Returns the Sharpe ratio based on a series of asset prices
        and risk-free asset prices. The calculation is based on the
        arithmetic mean of actual returns, as appears to be standard

        :param price_series - a pandas series or data frame of prices
        :param rf_series - a pandas series representing the total
        return/price series of the risk free rate
        :param scale_to_annualise - bool governing whether or not a
        scaling factor is applied to annualise the statistic

        Notes
        -------
        https://en.wikipedia.org/wiki/Sharpe_ratio
        """

        # excess_return = get_annualised_excess_return(price_series, rf_series)
        # vol = get_annual_vol(price_series)
        # sharpe_ratio = excess_return/vol

        returns = (price_series / price_series.shift(1)).dropna()
        returns_rf = (rf_series / rf_series.shift(1)).dropna()

        rel_returns = returns.subtract(returns_rf, axis=0)
        avg_excess = rel_returns.mean()
        vol = rel_returns.std()
        annualising_scaling = TSeriesHelper._get_annualisation_factor(price_series.index) if scale_to_annualise else 1

        sharpe_ratio = avg_excess * annualising_scaling / vol

        return sharpe_ratio

    @staticmethod
    def _get_downside_deviation(asset_returns: NDFrame,
                                threshold: int = 1):
        """
        Returns the downside annual vol - this is the vol of
        the returns relative to the threshold returns (capped
        at 0 at the upper bound)

        :param price_series - a pandas series or data frame of returns
        in the form (1+r)
        :param threshold - the threshold below which returns should be
        included in the calculation.  Default is a zero return (ie a
        threshold of 1).
        """

        returns_relative = asset_returns.subtract(threshold, axis=0)
        returns_clipped = returns_relative.where(returns_relative < 0, 0)

        returns_sqd = np.power(returns_clipped, 2)
        deviation = np.sum(returns_sqd) / len(returns_clipped)
        return np.sqrt(deviation)

    @staticmethod
    def _get_annualisation_factor(dates: np.array):
        """
        Helper function to generate the annualisation factor
        based on the available dates
        :param dates: a list of dates, representing the dates in
        time series
        :return:
        """

        time_delta = (dates[1:] - dates[0:-1])
        average_return_frequency = max(np.mean([x.days for x in time_delta]), 1)
        freq = TSeriesHelper._get_frequency_for_time_diff(average_return_frequency)
        annualising_scaling = np.sqrt(freq)

        return annualising_scaling

    @staticmethod
    def _get_frequency_for_time_diff(time_difference: float):
        """
        map the avg difference in time period to a
        annualsing constant: daily, weekly, monthly
        or yearly.  First, we check which range the
        mapped difference lies in, and then use this
        to map to a frequency
        :param time_difference: the average time difference between
        dates in a time series
        :return:
        """

        mapped_difference = None

        if tp.D_RANGE[0] <= time_difference <= tp.D_RANGE[1]:
            mapped_difference = tp.BD_PER_ANUM
        elif tp.W_RANGE[0] <= time_difference <= tp.W_RANGE[1]:
            mapped_difference = tp.W_PER_ANUM
        elif tp.M_RANGE[0] <= time_difference <= tp.M_RANGE[1]:
            mapped_difference = tp.M_PER_ANUM
        elif tp.Y_RANGE[0] <= time_difference <= tp.Y_RANGE[1]:
            mapped_difference = tp.Y_PER_ANUM

        if mapped_difference is None:
            raise ValueError("Frequency of time series is not recognised")
        else:
            return mapped_difference
