from tests import PySparkTestCase

import pyspark.sql.functions as f

from preprocess import Preprocess

import pandas as pd


class TestPreprocess(PySparkTestCase):

    def test_preprocess(self):

        df_recipe_info = self.spark.read.csv('tests/fixtures/preprocess/recipe_info.csv', header=True)

        preprocessor_all = Preprocess(df_labels=df_recipe_info, columns='all')

        df_preprocessed_all = preprocessor_all.preprocess()
        self.assertEqual(df_preprocessed_all.count(), df_recipe_info.count()-1)

        preprocessor_country = Preprocess(df_labels=df_recipe_info, columns=['country'])
        df_preprocessed_country = preprocessor_country.preprocess()
        self.assertEqual(df_preprocessed_country.count(), df_recipe_info.count() - 1)
        self.assertEqual(len(df_preprocessed_country.columns), 1+4)

    def test__rectify_country_labels(self):

        df_countries = self.spark.read.csv('tests/fixtures/preprocess/rectify_country_labels.csv', header=True)

        preprocessor = Preprocess(df_labels=df_countries, columns='all')
        df_rectified_country_labels = preprocessor._rectify_country_labels()

        check_recipe_2 = df_rectified_country_labels.filter(f.col('recipe_id') == '2').select('country').collect()[0][0]
        self.assertEqual(check_recipe_2, 'United States')

        check_recipe_3 = df_rectified_country_labels.filter(f.col('recipe_id') == '3').select('country').collect()[0][0]
        self.assertEqual(check_recipe_3, 'Israel')

        check_recipe_4 = df_rectified_country_labels.filter(f.col('recipe_id') == '4').select('country').collect()[0][0]
        self.assertEqual(check_recipe_4, 'South Korea')

        check_recipe_5 = df_rectified_country_labels.filter(f.col('recipe_id') == '5').select('country').collect()[0][0]
        self.assertEqual(check_recipe_5, 'South Korea')

        check_recipe_6 = df_rectified_country_labels.filter(f.col('recipe_id') == '6').select('country').collect()[0][0]
        self.assertEqual(check_recipe_6, 'United Kingdom')

        check_recipe_7 = df_rectified_country_labels.filter(f.col('recipe_id') == '7').select('country').collect()[0][0]
        self.assertEqual(check_recipe_7, 'United States')

        df_count_check = df_rectified_country_labels.where(f.col('country') == f.col('country_secondary'))
        self.assertEqual(df_count_check.count(), 10)

    def test__convert_prep_time(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor_all = Preprocess(df_labels=df_long, columns='all')
        df_converted_prep_time_all = preprocessor_all._convert_prep_time(df_long)

        recipe_2_check = df_converted_prep_time_all\
            .filter(f.col('recipe_id') == '2')\
            .select('prep_time')\
            .collect()[0][0]

        self.assertEqual(recipe_2_check, '60')

        preprocessor_country = Preprocess(df_labels=df_long, columns=['country'])
        df_converted_prep_time_country = preprocessor_country._convert_prep_time(df_long)

        self.assertEqual(len(df_converted_prep_time_country.columns), len(df_long.columns))

    def test__convert_column_argument(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor = Preprocess(df_labels=df_long, columns='all')

        self.assertEqual(len(df_long.columns)-1, len(preprocessor.columns))

    def test__remove_duplicate_indexes(self):

        df_duplicate_recipes = self.spark.read.csv('tests/fixtures/preprocess/duplicate_recipes.csv', header=True)

        preprocessor = Preprocess(df_labels=df_duplicate_recipes, columns='all')

        self.assertEqual(df_duplicate_recipes.count() - 1, preprocessor.df_labels.count())

    def test__check_is_list(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        Preprocess(df_labels=df_long, columns=['country', 'protein'])

        with self.assertRaises(AssertionError):
            Preprocess(df_labels=df_long, columns='protein')

    def test__remove_columns(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor_2_columns = Preprocess(df_labels=df_long, columns=['country', 'protein'])
        preprocessor_2_columns._remove_columns()

        self.assertEqual(len(preprocessor_2_columns.df_labels.columns), 1 + 2)

        preprocessor_all = Preprocess(df_labels=df_long, columns='all')
        preprocessor_all._remove_columns()

        self.assertEqual(len(preprocessor_all.df_labels.columns), 4)

    def test__convert_one_hot(self):

        df_long = self.spark.read.csv('tests/fixtures/preprocess/long.csv', header=True)

        preprocessor_all = Preprocess(df_labels=df_long, columns=['country', 'protein', 'prep_time'])
        df_one_hot_all = preprocessor_all._convert_to_one_hot(df_long)

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

        preprocessor_country = Preprocess(df_labels=df_long, columns=['country'])
        df_one_hot_country = preprocessor_country._convert_to_one_hot(df_long)

        self.assertEqual(len(df_one_hot_country.columns), 1+4+2)

    def test__convert_nas(self):

        df_nas = self.spark.read.csv('tests/fixtures/preprocess/nas.csv', header=True)

        preprocessor = Preprocess(df_labels=df_nas, columns=[''])

        df_converted_nas = preprocessor._convert_nas(df_nas)

        check_protein = [v[0] for v in df_converted_nas.select('protein').collect()]
        self.assertEqual(2, check_protein.count('protein_not_applicable'))

        check_protein_cut = [v[0] for v in df_converted_nas.select('protein_cut').collect()]
        self.assertEqual(2, check_protein_cut.count('protein_cut_not_applicable'))

        self.assertEqual(len(df_nas.columns), len(df_converted_nas.columns))

    def test__convert_columns_to_lower_case(self):

        df_upper_case = self.spark.read.csv('tests/fixtures/preprocess/upper_case.csv', header=True)

        preprocessor = Preprocess(df_labels=df_upper_case, columns=[''])

        df_lower_case = preprocessor._convert_columns_to_lower_case(df_upper_case)

        check_country = [v[0] for v in df_lower_case.select('country').collect()]
        for v in check_country:
            self.assertTrue(v.islower())

        check_diet_type = [v[0] for v in df_lower_case.select('diet_type').collect()]
        for v in check_diet_type:
            self.assertTrue(v.islower())

    def test__check_nulls_in_attribute_columns(self):

        df_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/nulls_attributes.csv', header=True)
        df_no_nulls_attributes = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_attributes.csv', header=True)

        Preprocess(df_labels=df_no_nulls_attributes, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocess(df_labels=df_nulls_attributes, columns=[''])

    def test__check_is_spark_data_frame(self):

        df_simple_table = self.spark.read.csv('tests/fixtures/preprocess/simple_table.csv', header=True)
        pd_df_simple_table = pd.read_csv('tests/fixtures/preprocess/simple_table.csv')

        Preprocess(df_labels=df_simple_table, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocess(df_labels=pd_df_simple_table, columns=[''])

    def test__check_nulls_in_index_column(self):

        df_nulls = self.spark.read.csv('tests/fixtures/preprocess/nulls_recipe_id.csv', header=True)
        df_no_nulls = self.spark.read.csv('tests/fixtures/preprocess/no_nulls_recipe_id.csv', header=True)

        Preprocess(df_labels=df_no_nulls, columns=[''])

        with self.assertRaises(AssertionError):
            Preprocess(df_labels=df_nulls, columns=[''])

    def test__replace_whitespaces(self):

        df_whitespaces = self.spark.read.csv('tests/fixtures/preprocess/replace_whitespaces.csv', header=True)

        preprocess = Preprocess(df_labels=df_whitespaces, columns=[''])
        df_no_whitespaces = preprocess._replace_whitespaces_with_underscores(df_whitespaces)

        check_country = [v[0] for v in df_no_whitespaces.select('country').collect()]
        check_dish_category = [v[0] for v in df_no_whitespaces.select('dish_category').collect()]

        for v in check_country:
            self.assertTrue(' ' not in v)
        for v in check_dish_category:
            self.assertTrue(' ' not in v)

        self.assertEqual(len(df_whitespaces.columns), len(df_no_whitespaces.columns))

