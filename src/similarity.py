from pyspark.sql import DataFrame


class Similarity(object):
    """
    Class to generate similarity scores.

    """

    def __init__(self, df_features):
        """

        :param df_features: spark data frame, contains labels in first column and int features in remaining

        """

        self.df_features = df_features

        self.check_is_spark_data_frame()

    def check_is_spark_data_frame(self):
        """
        Checks if df_recipe_info is a spark data frame.

        :return:
        """

        assert isinstance(self.df_features, DataFrame), '"df_recipe_info" is not a spark data frame.'


