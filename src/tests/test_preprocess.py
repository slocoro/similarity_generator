from tests import PySparkTestCase

from preprocess import Preprocessor

import pandas as pd


class TestPreprocessor(PySparkTestCase):

    def test_check_if_spark_data_frame(self):

        df_simple_table = self.spark.read.csv('tests/fixtures/preprocess/simple_table.csv', header=True)
        pd_df_simple_table = pd.read_csv('tests/fixtures/preprocess/simple_table.csv')

        Preprocessor(df_recipe_info=df_simple_table, columns='')

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=pd_df_simple_table, columns='')

    def test_check_if_recipe_id_contains_nulls(self):

        df_nulls = self.spark.read.csv('tests/fixtures/preprocess/nulls_recipe_id.csv', header=True)
        df_no_nulls = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_recipe_id.csv', header=True)

        Preprocessor(df_recipe_info=df_no_nulls, columns='')

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_nulls, columns='')


