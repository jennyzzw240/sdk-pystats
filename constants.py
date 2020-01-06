class TimePeriods:

    # time periods
    MONTH_TO_DATE = 'MTD'
    QUARTER_TO_DATE = 'QTD'
    YEAR_TO_DATE = 'YTD'
    THREE_MONTHS = '3m'
    TWELVE_MONTHS = 'Rolling 12M'
    SCA_INCEPTION = 'SCA_Inception_TD'
    PORTFOLIO_INCEPTION = 'Portfolio_Inception_TD'

    # numeric constants
    D_PER_ANUM = 365
    BD_PER_ANUM = 252
    W_PER_ANUM = 52
    M_PER_ANUM = 12
    Y_PER_ANUM = 1  # obviously....

    # ranges for mapping ann factor
    D_RANGE = (0.8, 1.5)
    W_RANGE = (6, 7)
    M_RANGE = (27, 32)
    Y_RANGE = (359, 371)

    # misc
    DATE_NAME = 'as_of_date'

    @classmethod
    def get_all_time_periods(cls):
        """
        returns a list containing all of the durations
        """

        output = [cls.MONTH_TO_DATE, cls.QUARTER_TO_DATE, cls.YEAR_TO_DATE, cls.THREE_MONTHS, cls.TWELVE_MONTHS,
                  cls.SCA_INCEPTION, cls.PORTFOLIO_INCEPTION]

        return output
