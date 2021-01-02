from tests import PySparkTestCase

import pyspark.sql.functions as f

from preprocess import Preprocessor

import pandas as pd


class TestPreprocessor(PySparkTestCase):

    def test_preprocess(self):

        df_recipe_info = self.spark.read.csv('tests/fixtures/preprocess/recipe_info.csv', header=True)

    def test_remove_duplicate_recipes(self):

        df_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/duplicate_recipes.csv', header=True)
        df_duplicate_recipes.show()

        preprocessor = Preprocessor(df_recipe_info=df_duplicate_recipes, columns='all')
        preprocessor.df_recipe_info.show()

        self.assertEqual(df_duplicate_recipes.count()-1, preprocessor.df_recipe_info.count())

    def test_check_is_list(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        Preprocessor(df_recipe_info=df_long, columns=['country', 'protein'])

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_long, columns='protein')

    def test_remove_columns(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor_2_columns = Preprocessor(df_recipe_info=df_long, columns=['country', 'protein'])
        preprocessor_2_columns.remove_columns()

        self.assertEqual(len(preprocessor_2_columns.df_recipe_info.columns), 1+2)

        preprocessor_all = Preprocessor(df_recipe_info=df_long, columns='all')
        preprocessor_all.remove_columns()

        self.assertEqual(len(preprocessor_all.df_recipe_info.columns), 4)

    def test_convert_one_hot(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor_all = Preprocessor(df_recipe_info=df_long, columns=['country', 'protein', 'prep_time'])
        df_one_hot_all = preprocessor_all.convert_to_one_hot(df_long)

        self.assertEqual(len(df_one_hot_all.columns), 13+1)

        columns_all = df_one_hot_all.columns

        country_columns = [col for col in columns_all if 'country' in col]
        df_country_check = df_one_hot_all.withColumn('total', sum(df_one_hot_all[col] for col in country_columns))
        country_max = df_country_check.select(f.max('total')).collect()[0][0]
        self.assertEqual(1, country_max)

        protein_columns = [col for col in columns_all if 'protein' in col]
        df_protein_check = df_one_hot_all.withColumn('total', sum(df_one_hot_all[col] for col in protein_columns))
        protein_max = df_protein_check.select(f.max('total')).collect()[0][0]
        self.assertEqual(1, protein_max)

        prep_time_columns = [col for col in columns_all if 'prep_time' in col]
        df_prep_time_check = df_one_hot_all.withColumn('total', sum(df_one_hot_all[col] for col in prep_time_columns))
        prep_time_max = df_prep_time_check.select(f.max('total')).collect()[0][0]
        self.assertEqual(1, prep_time_max)

        self.assertEqual(df_long.count(), df_one_hot_all.count())

        preprocessor_country = Preprocessor(df_recipe_info=df_long, columns=['country'])
        df_one_hot_country = preprocessor_country.convert_to_one_hot(df_long)

        self.assertEqual(len(df_one_hot_country.columns), 1+4+2)

    def test_convert_nas(self):

        df_nas = self.spark.read.csv('tests/fixtures/preprocess/nas.csv', header=True)

        preprocessor = Preprocessor(df_recipe_info=df_nas, columns=[''])

        df_converted_nas = preprocessor.convert_nas(df_nas)

        check_protein = [v[0] for v in df_converted_nas.select('protein').collect()]
        self.assertEqual(2, check_protein.count('protein_not_applicable'))

        check_protein_cut = [v[0] for v in df_converted_nas.select('protein_cut').collect()]
        self.assertEqual(2, check_protein_cut.count('protein_cut_not_applicable'))

        self.assertEqual(len(df_nas.columns), len(df_converted_nas.columns))

    def test_convert_columns_to_lower_case(self):

        df_upper_case = self.spark.read.csv('tests/fixtures/preprocess/upper_case.csv', header=True)

        preprocessor = Preprocessor(df_recipe_info=df_upper_case, columns=[''])

        df_lower_case = preprocessor.convert_columns_to_lower_case(df_upper_case)

        check_country = [v[0] for v in df_lower_case.select('country').collect()]
        for v in check_country:
            self.assertTrue(v.islower())

        check_diet_type = [v[0] for v in df_lower_case.select('diet_type').collect()]
        for v in check_diet_type:
            self.assertTrue(v.islower())

    def test_check_nulls_in_attribute_columns(self):

        df_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/nulls_attributes.csv', header=True)
        df_no_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_attributes.csv', header=True)

        Preprocessor(df_recipe_info=df_no_nulls_attributes, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_nulls_attributes, columns=[''])

    def test_check_is_spark_data_frame(self):

        df_simple_table = self.spark.read.csv('tests/fixtures/preprocess/simple_table.csv', header=True)
        pd_df_simple_table = pd.read_csv('tests/fixtures/preprocess/simple_table.csv')

        Preprocessor(df_recipe_info=df_simple_table, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=pd_df_simple_table, columns=[''])

    def test_check_nulls_in_recipe_id(self):

        df_nulls = self.spark.read.csv('tests/fixtures/preprocess/nulls_recipe_id.csv', header=True)
        df_no_nulls = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_recipe_id.csv', header=True)

        Preprocessor(df_recipe_info=df_no_nulls, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_nulls, columns=[''])

    def test_check_no_duplicate_recipes(self):

        df_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/duplicate_recipes.csv', header=True)
        df_no_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/no_duplicate_recipes.csv', header=True)

        with self.assertRaises(AssertionError):
            Preprocessor(df_recipe_info=df_duplicate_recipes, columns=[''])

        Preprocessor(df_recipe_info=df_no_duplicate_recipes, columns=[''])

    def test_replace_whitespaces(self):

        df_whitespaces = self.spark.read.csv('tests/fixtures/preprocess/replace_whitespaces.csv', header=True)

        preprocess = Preprocessor(df_recipe_info=df_whitespaces, columns=[''])
        df_no_whitespaces = preprocess.replace_whitespaces_with_underscores()

        check_country = [v[0] for v in df_no_whitespaces.select('country').collect()]
        check_dish_category = [v[0] for v in df_no_whitespaces.select('dish_category').collect()]

        for v in check_country:
            self.assertTrue(' ' not in v)
        for v in check_dish_category:
            self.assertTrue(' ' not in v)

        self.assertEqual(len(df_whitespaces.columns), len(df_no_whitespaces.columns))

