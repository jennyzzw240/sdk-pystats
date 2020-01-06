import unittest
from datetime import datetime
import pandas as pd
from pystats.constants import TimePeriods as tp
from pystats.risk_return import TSeriesHelper
from pystats.tests.helper_functions import third_friday_of_month_dates


class TestRiskReturn(unittest.TestCase):

    def setUp(self):
        start_date = datetime(2016, 4, 1)
        end_date = datetime(2018, 2, 28)
        trading_days = third_friday_of_month_dates(start_date, end_date)

        test_dict = {'nky': [100, 101, 102, 102, 103, 102, 100, 99, 98, 97, 98, 100, 102, 104, 103, 105, 104, 107, 106,
                             105, 110, 105, 100],
                     'spx': [100, 99, 97, 96, 90, 86, 87, 88, 80, 84, 86, 85, 90, 91, 92, 94, 95, 97, 91, 93, 98, 101,
                             102],
                     'rf': [100 + 0.05 * n for n in range(0, 23)]}

        self.price_data = pd.DataFrame(test_dict, index=list(trading_days))

    def test_annual_vol(self):

        vols = TSeriesHelper.get_annualised_vol(self.price_data)

        self.assertAlmostEqual(vols['nky'], 0.07611, places=5)
        self.assertAlmostEqual(vols['spx'], 0.13227, places=5)

    def test_calculate_annualised_return(self):

        returns = TSeriesHelper.get_annualised_returns(self.price_data)

        self.assertAlmostEqual(returns['nky'], 0)
        self.assertAlmostEqual(returns['spx'], 0.01081, places=5)

    def test_get_annualised_excess_return(self):

        market_prices = self.price_data[['nky', 'spx']]
        risk_free_prices = self.price_data['rf']

        result = TSeriesHelper.get_annualised_excess_return(market_prices, risk_free_prices)

        self.assertAlmostEqual(result['nky'], -0.00596, places=5)
        self.assertAlmostEqual(result['spx'], 0.00485, places=5)

    def test_get_sharpe_ratio_scaled(self):

        market_prices = self.price_data[['nky', 'spx']]
        risk_free_prices = self.price_data['rf']

        result = TSeriesHelper.get_sharpe_ratio_scaled(market_prices, risk_free_prices)

        self.assertAlmostEqual(result['nky'], -0.03964, places=5)
        self.assertAlmostEqual(result['spx'], 0.10120, places=5)

    def test_get_sharpe_ratio_unscaled(self):

        market_prices = self.price_data[['nky', 'spx']]
        risk_free_prices = self.price_data['rf']

        result = TSeriesHelper.get_sharpe_ratio_unscaled(market_prices, risk_free_prices)

        self.assertAlmostEqual(result['nky'], -0.01144, places=5)
        self.assertAlmostEqual(result['spx'], 0.029213, places=5)

    def test_get_downside_deviation(self):

        market_prices = self.price_data[['nky', 'spx']]
        market_returns = (market_prices / market_prices.shift(1)).dropna()
        risk_free_prices = self.price_data['rf']
        risk_free_returns = (risk_free_prices / risk_free_prices.shift(1)).dropna()

        result = TSeriesHelper._get_downside_deviation(market_returns, risk_free_returns)

        self.assertAlmostEqual(result['nky'], 0.01605, places=5)
        self.assertAlmostEqual(result['spx'], 0.02941, places=5)

    def test_get_sortino_ratio_scaled(self):

        market_prices = self.price_data[['nky', 'spx']]
        risk_free_prices = self.price_data['rf']

        result = TSeriesHelper.get_sortino_ratio_scaled(market_prices, risk_free_prices)

        self.assertAlmostEqual(result['nky'], -0.05536, places=5)
        self.assertAlmostEqual(result['spx'], 0.13248, places=5)

    def test_get_sortino_ratio_unscaled(self):

        market_prices = self.price_data[['nky', 'spx']]
        risk_free_prices = self.price_data['rf']

        result = TSeriesHelper.get_sortino_ratio_unscaled(market_prices, risk_free_prices)

        self.assertAlmostEqual(result['nky'], -0.01598, places=5)
        self.assertAlmostEqual(result['spx'], 0.03825, places=5)

    def test_get_ann_const_from_time_diff(self):

        diff_1 = 1.1
        diff_2 = 6.9
        diff_3 = 28
        diff_4 = 361
        diff_5 = 10

        self.assertEqual(tp.BD_PER_ANUM, TSeriesHelper._get_frequency_for_time_diff(diff_1))
        self.assertEqual(tp.W_PER_ANUM, TSeriesHelper._get_frequency_for_time_diff(diff_2))
        self.assertEqual(tp.M_PER_ANUM, TSeriesHelper._get_frequency_for_time_diff(diff_3))
        self.assertEqual(tp.Y_PER_ANUM, TSeriesHelper._get_frequency_for_time_diff(diff_4))

        with(self.assertRaises(ValueError)):
            self.assertEqual(tp.BD_PER_ANUM, TSeriesHelper._get_frequency_for_time_diff(diff_5))

    def test_price_to_return(self):

        start_date = pd.datetime(2018, 1, 1)
        end_date = pd.datetime(2018, 1, 12)
        dates = pd.date_range(start_date, end_date, freq='B')

        test_dict = {'test_1': [100, 102, 104, 106, 102, 98, 100, 103, 107, 110],
                     'test_2': [21, 22, 25, 27, 21, 24, 24, 23, 22, 21]}

        test_df = pd.DataFrame(test_dict, index=dates)

        result = TSeriesHelper.price_to_return(test_df)

        expected_dates = list(dates)[1:]
        expected_dict = {'test_1': [0.0200, 0.019608, 0.019231, -0.037736, -0.0392157,
                                    0.020408, 0.0300, 0.038835, 0.028037],
                         'test_2': [0.047619, 0.136364, 0.0800, -0.222222, 0.142857,
                                    0, -0.041667, -0.0434782, -0.045454]}
        expected_df = pd.DataFrame(expected_dict, index=expected_dates, )

        pd.testing.assert_frame_equal(result, expected_df, check_less_precise=4)

    def test_return_to_price(self):

        start_date = pd.datetime(2017, 2, 1)
        end_date = pd.datetime(2017, 2, 5)
        dates = pd.date_range(start_date, end_date, freq='D')

        test_dict = {'test_1': [0.01, 0.02, 0.02, 0.01, 0.03],
                     'test_2': [-0.01, -0.04, -0.02, 0.05, 0.05]}
        test_df = pd.DataFrame(test_dict, index=dates)

        result = TSeriesHelper.return_to_price(test_df)

        expected_dict = {'test_1': [101, 103.02, 105.0804, 106.1312, 109.3151],
                         'test_2': [99, 95.04, 93.1392, 97.7962, 102.6860]}
        expected_df = pd.DataFrame(expected_dict, index=dates)

        pd.testing.assert_frame_equal(result, expected_df)
