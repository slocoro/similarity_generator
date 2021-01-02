from pyspark.sql import DataFrame


class Preprocessor(object):
    """
    Prepare data for similarity calculation.

    """

    def __init__(self, df_recipe_info, columns):
        """

        :param df_recipe_info: spark data frame
        :param columns: list of string, columns to use for similarity calculation
        """

        self.df_recipe_info = df_recipe_info
        self.columns = columns

        self.check_if_spark_data_frame()

    def check_if_spark_data_frame(self):
        """
        Checks if df_recipe_info is a spark data frame.

        :return:
        """

        assert isinstance(self.df_recipe_info, DataFrame)
