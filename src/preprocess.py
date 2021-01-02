from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import Window


class Preprocessor(object):
    """
    Prepare data for similarity calculation.

    """

    def __init__(self, df_recipe_info, columns):
        """
        Performs the following assumption checks/manipulations during initialization:
            - checks if "df_recipe_info" is a spark data frame
            - checks "columns" is a list or "all"
            - convert "columns" to list of containing all columns from "df_recipe_id"
            - checks nulls in "recipe_id"
            - removes duplicates from "recipe_id"
            - checks if attribute columns contain nulls

        :param df_recipe_info: spark data frame
        :param columns: list of string, columns to use for similarity calculation
        """

        self.df_recipe_info = df_recipe_info
        self.columns = columns

        self.check_is_spark_data_frame()
        self.check_is_list()
        self.convert_column_argument()
        self.check_nulls_in_recipe_id()
        self.remove_duplicate_recipes()
        self.check_nulls_in_attribute_columns()

    def convert_column_argument(self):
        """
        Converts column argument to list of columns names in df_recipe_info (without recipe_id).

        :return:
        """

        if self.columns == 'all':
            self.columns = [col for col in self.df_recipe_info.columns if col != 'recipe_id']

    def remove_duplicate_recipes(self):
        """
        Removes duplicate recipes by randomly selecting one if duplicated.

        :return:
        """

        window = Window \
            .partitionBy(['recipe_id']) \
            .orderBy(f.rand())

        self.df_recipe_info = self.df_recipe_info\
            .withColumn('rn', f.row_number().over(window))\
            .filter(f.col('rn') == 1)\
            .drop('rn')

    def check_is_list(self):
        """
        Checks "columns" is a list.

        :return:
        """

        if self.columns is not 'all':
            assert isinstance(self.columns, list), '"columns" has to be a list.'

    def check_is_spark_data_frame(self):
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

    def preprocess(self):
        """
        Preprocess recipes data.

        :return: spark data frame
        """

        self.remove_columns()

        df_rectified_country_labels = self.rectify_country_labels()
        df_no_whitespaces = self.replace_whitespaces_with_underscores(df_rectified_country_labels)
        df_lower_case = self.convert_columns_to_lower_case(df_no_whitespaces)
        df_converted_nas = self.convert_nas(df_lower_case)
        df_converted_prep_time = self.convert_prep_time(df_converted_nas)
        df_one_hot = self.convert_to_one_hot(df_converted_prep_time)

        return df_one_hot

    def remove_columns(self):
        """
        Removes columns not in self.columns

        :return:
        """

        if self.columns == 'all':
            pass
        else:
            self.df_recipe_info = self.df_recipe_info.select(['recipe_id'] + self.columns)

    def rectify_country_labels(self):
        """
        Rectifies inconsistent country labels.

        :return: spark data frame
        """

        country_columns = [col for col in self.columns if 'country' in col]
        df_rectified_country_labels = self.df_recipe_info

        if country_columns:
            for country in country_columns:
                df_rectified_country_labels = df_rectified_country_labels\
                    .withColumn(country, f.regexp_replace(country,
                                                          'United States of America \(USA\)',
                                                          'United States'))
                df_rectified_country_labels = df_rectified_country_labels\
                    .withColumn(country, f.regexp_replace(country,
                                                          'Israel and the Occupied Territories',
                                                          'Israel'))
                df_rectified_country_labels = df_rectified_country_labels\
                    .withColumn(country, f.regexp_replace(country,
                                                          'Korea, Republic of \(South Korea\)',
                                                          'South Korea'))

                df_rectified_country_labels = df_rectified_country_labels\
                    .withColumn(country, f.regexp_replace(country,
                                                          'Korea, Democratic Republic of \(North Korea\)',
                                                          'South Korea'))

                df_rectified_country_labels = df_rectified_country_labels\
                    .withColumn(country, f.regexp_replace(country,
                                                          'Great Britain',
                                                          'United Kingdom'))

        return df_rectified_country_labels

    @staticmethod
    def replace_whitespaces_with_underscores(df_rectified_country_labels):
        """
        Replaces whitespaces with underscores in every column except "recipe_id"

        :return:
        """

        columns_to_process = [col for col in df_rectified_country_labels.columns if col != 'recipe_id']

        df_withspaces = df_rectified_country_labels

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

    def convert_prep_time(self, df_converted_nas):
        """
        Converts prep times in ranges to upper bound of range.

        :param df_converted_nas: spark data frame
        :return: spark data frame
        """

        if 'prep_time' in self.columns:
            convert_prep_time = f.udf(lambda x: x.split('-')[-1], StringType())

            df_converted_nas = df_converted_nas.withColumn('prep_time_copy', f.col('prep_time'))

            df_converted_prep_time = df_converted_nas.withColumn('prep_time', convert_prep_time(df_converted_nas.prep_time_copy))
            df_converted_prep_time = df_converted_prep_time.drop('prep_time_copy')

            return df_converted_prep_time
        else:
            return df_converted_nas

    @staticmethod
    def convert_nas(df_lower_case):
        """
        Converts "#n/a" to column_name+not_applicable.

        :param df_lower_case: spark data frame
        :return: spark data frame
        """

        columns_to_process = [col for col in df_lower_case.columns if col != 'recipe_id']

        df_converted_nas = df_lower_case

        for col in columns_to_process:
            df_converted_nas = df_converted_nas.withColumn(col, f.regexp_replace(col, '#n/a', col+'_not_applicable'))

        return df_converted_nas

    def convert_to_one_hot(self, df_lower_case):
        """
        Converts recipes description data to one hot using columns attribute.

        :param df_lower_case: spark data frame
        :return: spark data frame
        """

        df_one_hot = df_lower_case

        for col in self.columns:
            unique_labels = [v[0] for v in df_lower_case.select(col).distinct().collect()]

            for label in unique_labels:
                df_one_hot = df_one_hot\
                    .withColumn(col+'_'+label, f.when(f.col(col) == label, 1).otherwise(0))

            df_one_hot = df_one_hot.drop(col)

        return df_one_hot

