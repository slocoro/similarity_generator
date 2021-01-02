from tests import PySparkTestCase

from preprocess import Preprocessor

import pandas as pd


class TestPreprocessor(PySparkTestCase):

    def test_check_nulls_in_attribute_columns(self):

        df_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/nulls_attributes.csv', header=True)
        df_no_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_attributes.csv', header=True)

        Preprocessor(df_recipe_info=df_no_nulls_attributes, columns='')

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_nulls_attributes, columns='')

    def test_check_if_spark_data_frame(self):

        df_simple_table = self.spark.read.csv('tests/fixtures/preprocess/simple_table.csv', header=True)
        pd_df_simple_table = pd.read_csv('tests/fixtures/preprocess/simple_table.csv')

        Preprocessor(df_recipe_info=df_simple_table, columns='')

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=pd_df_simple_table, columns='')

    def test_check_nulls_in_recipe_id(self):

        df_nulls = self.spark.read.csv('tests/fixtures/preprocess/nulls_recipe_id.csv', header=True)
        df_no_nulls = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_recipe_id.csv', header=True)

        Preprocessor(df_recipe_info=df_no_nulls, columns='')

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_nulls, columns='')

    def test_check_no_duplicate_recipes(self):

        df_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/duplicate_recipes.csv', header=True)
        df_no_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/no_duplicate_recipes.csv', header=True)

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_duplicate_recipes, columns='')

        Preprocessor(df_recipe_info=df_no_duplicate_recipes, columns='')

    def test_replace_whitespaces(self):

        df_whitespaces = self.spark.read.csv('tests/fixtures/preprocess/replace_whitespaces.csv', header=True)

        preprocess = Preprocessor(df_recipe_info=df_whitespaces, columns='')
        df_no_whitespaces = preprocess.replace_whitespaces_with_underscores()

        check_country = [v[0] for v in df_no_whitespaces.select('country').collect()]
        check_dish_category = [v[0] for v in df_no_whitespaces.select('dish_category').collect()]

        for v in check_country:
            self.assertTrue(' ' not in v)
        for v in check_dish_category:
            self.assertTrue(' ' not in v)

        self.assertEqual(len(df_whitespaces.columns), len(df_no_whitespaces.columns))

