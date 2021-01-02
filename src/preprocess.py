from pyspark.sql import DataFrame
import pyspark.sql.functions as f


class Preprocessor(object):
    """
    Prepare data for similarity calculation.

    """

    def __init__(self, df_recipe_info, columns):
        """
        Performs the following assumption checks during initialization:
            - checks if df_recipe_info is a spark data frame
            - checks if "recipe_id" contains nulls
            - checks if "recipe_id" contains duplicate
            - checks if attribute columns contain nulls

        :param df_recipe_info: spark data frame
        :param columns: list of string, columns to use for similarity calculation
        """

        self.df_recipe_info = df_recipe_info
        self.columns = columns

        self.check_if_spark_data_frame()
        self.check_nulls_in_recipe_id()
        self.check_no_duplicate_recipes()
        self.check_nulls_in_attribute_columns()

    def check_no_duplicate_recipes(self):
        """
        Checks there are no duplicates in the "recipe_id" column.

        :return:
        """

        row_count = self.df_recipe_info.count()
        recipe_id_count = self.df_recipe_info.select('recipe_id').distinct().count()

        assert row_count == recipe_id_count, 'There are duplicates in "recipe_id".'

    def check_if_spark_data_frame(self):
        """
        Checks if df_recipe_info is a spark data frame.

        :return:
        """

        assert isinstance(self.df_recipe_info, DataFrame), '"df_recipe_info" is not a spark data frame.'

    def check_nulls_in_recipe_id(self):
        """
        Checks if column "recipe_id" contains nulls.

        :return:
        """

        null_count = self.df_recipe_info.filter(f.col('recipe_id').isNull()).count()
        assert null_count == 0, \
            f'There are {null_count} null(s) in the "recipe_id" column in "df_recipe_info" when no nulls are allowed.'

    def check_nulls_in_attribute_columns(self):
        """
        Checks if nulls in attribute columns.

        :return:
        """

        columns_to_check = [col for col in self.df_recipe_info.columns if col != 'recipe_id']
        row_count = self.df_recipe_info.count()

        for col in columns_to_check:
            col_count = self.df_recipe_info.filter(f.col(col).isNotNull()).select(col).count()

            assert col_count == row_count, f'There are null(s) in "{col}".'

    def replace_whitespaces_with_underscores(self):
        """
        Replaces whitespaces with underscores in every column except "recipe_id"

        :return:
        """

        columns_to_process = [col for col in self.df_recipe_info.columns if col != 'recipe_id']

        df_withspaces = self.df_recipe_info

        for col in columns_to_process:
            df_withspaces = df_withspaces.withColumn(col, f.regexp_replace(col, ' ', '_'))

        df_no_whitespaces = df_withspaces

        return df_no_whitespaces

    @staticmethod
    def convert_columns_to_lower_case(df_no_whitspaces):
        """
        Converts all attriute columns to lower case.

        :return: spark data frame
        """

        columns_to_process = [col for col in df_no_whitspaces.columns if col != 'recipe_id']

        df_lower_case = df_no_whitspaces

        for col in columns_to_process:
            df_lower_case = df_lower_case.withColumn(col, f.lower(f.col(col)))

        return df_lower_case



