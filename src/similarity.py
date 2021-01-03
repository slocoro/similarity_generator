from pyspark.sql import DataFrame
import pyspark.sql.functions as f


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
        self.check_nulls_in_feature_columns()
        self.check_is_numerical_data()

    def check_is_spark_data_frame(self):
        """
        Checks if df_recipe_info is a spark data frame.

        :return:
        """

        assert isinstance(self.df_features, DataFrame), '"df_recipe_info" is not a spark data frame.'

    def check_is_numerical_data(self):
        """
        Checks of feature columns are numerical.

        :return:
        """

        columns_to_check = [col[1] for col in self.df_features.dtypes if 'id' not in col[0]]

        assert all((col == 'int' or col == 'double') for col in columns_to_check)

    def check_nulls_in_feature_columns(self):
        """
        Checks there are no nulls in the feature columns.

        :return:
        """

        columns_to_check = [col for col in self.df_features.columns if col != 'recipe_id']
        row_count = self.df_features.count()

        for col in columns_to_check:
            col_count = self.df_features.filter(f.col(col).isNotNull()).select(col).count()

            assert col_count == row_count, f'There are null(s) in "{col}".'

