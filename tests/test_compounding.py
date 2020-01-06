import unittest
import pandas as pd
import numpy as np
from pystats.compounding import _get_cum_prod_to_last_date, \
    _adjust_cum_prod_by_duration, _get_annualised_conts, get_compound_conts_to_pfo_annualised, \
    remove_undefined_contributions
from pystats.compounding import CompoundingTime
from pystats.constants import TimePeriods as tp
from unittest import mock


class TestCompounding(unittest.TestCase):
    def test_get_compounding_dates(self):
        date_1 = pd.datetime(2017, 12, 31)
        date_2 = pd.datetime(2016, 12, 31)
        date_3 = pd.datetime(2017, 5, 15)
        date_4 = pd.datetime(2017, 6, 30)
        date_5 = pd.datetime(2018, 1, 20)

        ct = CompoundingTime()
        ct_2 = CompoundingTime(optional_dates={tp.SCA_INCEPTION: date_4})

        res_1 = ct.get_compounding_dates(date_1, date_2)
        res_2 = ct.get_compounding_dates(date_3, date_2)
        res_3 = ct.get_compounding_dates(date_2, date_3)
        res_4 = ct.get_compounding_dates(date_2, date_2)
        res_5 = ct.get_compounding_dates(date_4, date_4)
        res_6 = ct_2.get_compounding_dates(date_1, date_2)
        res_7 = ct.get_compounding_dates(date_5, date_1)

        ans_1 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2017, 12, 31),
                      tp.QUARTER_TO_DATE: pd.Timestamp(2017, 10, 31),
                      tp.YEAR_TO_DATE: pd.Timestamp(2017, 1, 31),
                      tp.THREE_MONTHS: pd.Timestamp(2017, 10, 31),
                      tp.TWELVE_MONTHS: pd.Timestamp(2017, 1, 31)})

        ans_2 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2017, 5, 15),
                      tp.QUARTER_TO_DATE: pd.Timestamp(2017, 4, 30),
                      tp.YEAR_TO_DATE: pd.Timestamp(2017, 1, 31),
                      tp.THREE_MONTHS: pd.Timestamp(2017, 3, 31)})

        ans_3 = dict()

        ans_4 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2016, 12, 31)})

        ans_5 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2017, 6, 30)})

        ans_6 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2017, 12, 31),
                      tp.QUARTER_TO_DATE: pd.Timestamp(2017, 10, 31),
                      tp.YEAR_TO_DATE: pd.Timestamp(2017, 1, 31),
                      tp.THREE_MONTHS: pd.Timestamp(2017, 10, 31),
                      tp.TWELVE_MONTHS: pd.Timestamp(2017, 1, 31),
                      tp.SCA_INCEPTION: pd.Timestamp(2017, 6, 30)})

        ans_7 = dict({tp.MONTH_TO_DATE: pd.Timestamp(2018, 1, 20),
                      tp.QUARTER_TO_DATE: pd.Timestamp(2018, 1, 20),
                      tp.YEAR_TO_DATE: pd.Timestamp(2018, 1, 20)})

        self.assertEqual(res_1, ans_1)
        self.assertEqual(res_2, ans_2)
        self.assertEqual(res_3, ans_3)
        self.assertEqual(res_4, ans_4)
        self.assertEqual(res_5, ans_5)
        self.assertEqual(res_6, ans_6)
        self.assertEqual(res_7, ans_7)

    def test_get_cum_prod_to_last_date(self):
        df_1 = pd.DataFrame(np.array([[1, 2, 3]] * 3).T)
        df_2 = pd.DataFrame(np.array([[1, 2, 3, 4]]))
        df_3 = pd.DataFrame(np.array([1, 2, 3, 4]).T)
        df_4 = pd.DataFrame(np.array([[1, 1], [1, 0]]))

        res_1 = _get_cum_prod_to_last_date(df_1)
        res_2 = _get_cum_prod_to_last_date(df_2)
        res_3 = _get_cum_prod_to_last_date(df_3)

        ans_1 = pd.DataFrame(np.array([[6, 3, 1]] * 3).T)
        ans_2 = pd.DataFrame(np.array([[1, 1, 1, 1]]))
        ans_3 = pd.DataFrame(np.array([24, 12, 4, 1]))

        np.testing.assert_array_equal(res_1, ans_1)
        np.testing.assert_array_equal(res_2, ans_2)
        np.testing.assert_array_equal(res_3, ans_3)

        self.assertRaises(ValueError, _get_cum_prod_to_last_date, df_4)

    def test_adjust_cum_prod_by_duration(self):
        start_date = pd.datetime(2017, 5, 31)
        mid_date = pd.datetime(2017, 12, 31)
        end_date = pd.datetime(2018, 4, 30)

        ct = CompoundingTime({tp.SCA_INCEPTION: pd.datetime(2017, 6, 30)})

        dates_1 = ct.get_compounding_dates(end_date, start_date)
        dates_2 = ct.get_compounding_dates(end_date, mid_date)
        dates_3 = ct.get_compounding_dates(mid_date, start_date)

        res_rng_1 = np.vstack(
            [[1.039046, 1.028758, 1.039150, 1.018774, 1.008687, 0.988909, 0.960106, 0.9797, 1.01, 1.01, 1.01, 1]] * 6).T
        res_rng_2 = res_rng_1[-5:, :4]
        res_rng_3 = res_rng_1[:8, :4] / 0.9797

        res_df_1 = pd.DataFrame(res_rng_1, columns=dates_1.keys(),
                                index=pd.date_range(start=start_date, end=end_date, freq='M'))
        res_df_2 = pd.DataFrame(res_rng_2, columns=dates_2.keys(),
                                index=pd.date_range(start=mid_date, end=end_date, freq='M'))
        res_df_3 = pd.DataFrame(res_rng_3, columns=dates_3.keys(),
                                index=pd.date_range(start=start_date, end=mid_date, freq='M'))

        res_1 = _adjust_cum_prod_by_duration(res_df_1, dates_1)
        res_2 = _adjust_cum_prod_by_duration(res_df_2, dates_2)
        res_3 = _adjust_cum_prod_by_duration(res_df_3, dates_3)

        ans_rng_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1.01, 1.01, 1.01, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.01, 1.01, 1],
                              [1.039046, 1.028758, 1.039150, 1.018774, 1.008687, 0.988909, 0.960106, 0.979700, 1.01,
                               1.01, 1.01, 1],
                              [0, 1.028758, 1.039150, 1.018774, 1.008687, 0.988909, 0.960106, 0.979700, 1.01, 1.01,
                               1.01, 1]])

        ans_rng_2 = np.array([[0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 1.01, 1.01, 1.01, 1],
                              [0, 0, 1.01, 1.01, 1]])

        ans_rng_3 = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1.009400, 0.98, 1],
                              [0, 0, 0, 0, 0, 1.009400, 0.98, 1],
                              [0, 1.050075, 1.060682, 1.039884, 1.029588, 1.009400, 0.98, 1]])

        ans_1 = pd.DataFrame(ans_rng_1.T, columns=dates_1.keys(),
                             index=pd.date_range(start=start_date, end=end_date, freq='M'))
        ans_2 = pd.DataFrame(ans_rng_2.T, columns=dates_2.keys(),
                             index=pd.date_range(start=mid_date, end=end_date, freq='M'))
        ans_3 = pd.DataFrame(ans_rng_3.T, columns=dates_3.keys(),
                             index=pd.date_range(start=start_date, end=mid_date, freq='M'))

        np.testing.assert_array_almost_equal(res_1, ans_1, decimal=6)
        np.testing.assert_array_almost_equal(res_2, ans_2, decimal=6)
        np.testing.assert_array_almost_equal(res_3, ans_3, decimal=6)

    def test_compound_contribution_to_portfolio(self):
        # set up portfolio with 5 month history of three assets,
        # where thesis_3 only enters the portfolio in the
        # third month

        start_date = pd.datetime(2017, 10, 31)
        end_date = pd.datetime(2018, 2, 15)
        dates = list(pd.date_range(start_date, end_date, freq='M'))
        dates.append(pd.datetime(2018, 2, 15))

        sca_inception_date = pd.datetime(2017, 12, 31)
        portfolio_inception_date = pd.datetime(2017, 10, 31)

        input_dict = {'thesis_1': [0.01] * 5,
                      'thesis_2': [0.01, -0.01, 0.01, 0.01, 0.01],
                      'thesis_3': [0, 0, 0.02, 0.02, 0.01]}

        input_df = pd.DataFrame(input_dict, index=dates)

        optional_dates = {tp.SCA_INCEPTION: sca_inception_date,
                          tp.PORTFOLIO_INCEPTION: portfolio_inception_date}
        result = get_compound_conts_to_pfo_annualised(input_df, optional_dates, None)

        # set up expected result

        expected_index = ['thesis_1'] * 5 + ['thesis_2'] * 5 + ['thesis_3'] * 5

        expected_dict = {tp.DATE_NAME: dates[::-1] * 3,
                         tp.MONTH_TO_DATE: [0.01] * 5 + [0.01, 0.01, 0.01, -0.01, 0.01] + [0.01, 0.02, 0.02, 0, 0],
                         tp.QUARTER_TO_DATE: [0.0203, 0.01, 0.0308, 0.02, 0.01] + [0.0203, 0.01, 0.01, 0, 0.01] +
                                             [0.0306, 0.02, 0.02, 0, 0],
                         tp.YEAR_TO_DATE: [0.0203, 0.01, None, None, None] + [0.0203, 0.01, None, None, None]
                         + [0.0306, 0.02, None, None, None],
                         tp.THREE_MONTHS: [0.03101, 0.03122, 0.0308, None, None] + [0.03101, 0.00958, 0.01, None, None]
                         + [0.05202, 0.0408, 0.02, None, None],
                         tp.SCA_INCEPTION: [0.03101, 0.0204, 0.01, None, None] + [0.03101, 0.0204, 0.01, None, None]
                         + [0.05202, 0.0408, 0.02, None, None],
                         tp.PORTFOLIO_INCEPTION: [0.05329, 0.04203, 0.0308, 0.02, 0.01] + [0.03101, 0.0204, 0.01, 0,
                         0.01] + [0.05202, 0.0408, 0.02, 0, 0]}

        expected_df = pd.DataFrame(expected_dict, index=expected_index)
        expected_df = expected_df[[tp.DATE_NAME, tp.THREE_MONTHS, tp.MONTH_TO_DATE, tp.PORTFOLIO_INCEPTION,
                                   tp.QUARTER_TO_DATE, tp.SCA_INCEPTION, tp.YEAR_TO_DATE]]

        # sort so that both in same order for comparison

        expected_df = expected_df.set_index(tp.DATE_NAME, append=True).sort_index(axis=0).reset_index(level=1)
        result = result.set_index(tp.DATE_NAME, append=True).sort_index(axis=0).reset_index(level=1)

        pd.testing.assert_frame_equal(expected_df, result, check_less_precise=True)

    def test_get_annualised_optional(self):
        # set up compounded returns from
        # end march 17 and annualise

        start_date = pd.datetime(2017, 1, 31)
        dates = pd.date_range(start_date, periods=18, freq='M')

        ct = CompoundingTime({'test_date': pd.datetime(2017, 3, 31)})

        test_dict = {tp.DATE_NAME: dates,
                     'test_date': [None] * 2 + [0.01 * s for s in range(1, 17)]}

        test_df = pd.DataFrame(test_dict)

        result = _get_annualised_conts(test_df, ct)

        expected_dict = {tp.DATE_NAME: dates,
                         'test_date': [None] * 2 + [0.01 * s for s in range(1, 13)]
                         + [0.11943, 0.11886, 0.11830, 0.11775]}
        expected_df = pd.DataFrame(expected_dict)

        pd.testing.assert_frame_equal(result, expected_df, check_less_precise=4)

    def test_get_annualised_fixed(self):
        # set up a series of 3 year
        # returns which need to be annualised

        start_date = pd.datetime(2018, 1, 1)
        dates = pd.date_range(start_date, periods=5, freq='M')

        ct = CompoundingTime()
        three_yr_ann = {'three yrs': 36}
        ct.get_fixed_annualisations = mock.MagicMock(return_value=three_yr_ann)

        test_dict = {tp.DATE_NAME: dates,
                     'three yrs': [None, 0.1, -0.08, -0.04, 0]}
        test_df = pd.DataFrame(test_dict)

        result = _get_annualised_conts(test_df, ct)

        expected_dict = {tp.DATE_NAME: dates,
                         'three yrs': [None, 0.03228, -0.02741, -0.013515, 0]}
        expected_df = pd.DataFrame(expected_dict)

        pd.testing.assert_frame_equal(result, expected_df, check_less_precise=4)

    def test_remove_undefined_contributions(self):
        # arrange

        test_dict = {"Dates": [pd.datetime(2018, 1, 31)] * 5 +
                              [pd.datetime(2018, 2, 28)] * 5,
                     "Instrument": ["instr_" + str(i) for i in range(5)] * 2,
                     "Performance": [None, 0.02, None, 0.01, None] +
                                    [0, 0.01, 0.02, 0.01, 0.02]}
        test_df = pd.DataFrame(test_dict)

        expected_dict = {"Dates": [pd.datetime(2018, 1, 31)] * 2 +
                                  [pd.datetime(2018, 2, 28)] * 4,
                         "Instrument": ["instr_1", "instr_3"] +
                                       ["instr_1", "instr_2", "instr_3", "instr_4"],
                         "Performance": [0.02, 0.01] + [0.01, 0.02, 0.01, 0.02]}
        expected_df = pd.DataFrame(expected_dict)

        # act

        result = remove_undefined_contributions(test_df, "Performance")

        # assert

        expected_df = expected_df[result.columns]  # reorder for comparison
        result = result.reset_index(drop=True)  # reset index for comparison
        pd.testing.assert_frame_equal(expected_df, result)
