from tests import PySparkTestCase

import pyspark.sql.functions as f
from pyspark.sql.types import *

from similarity import Similarity

import pandas as pd


class TestSimiarity(PySparkTestCase):

    def test_check_is_spark_data_frame(self):

        df_simple_table = self.spark.read.csv('tests/fixtures/similarity/simple_table.csv', header=True)
        pd_df_simple_table = pd.read_csv('tests/fixtures/similarity/simple_table.csv')

        Similarity(df_features=df_simple_table)

        with self.assertRaises(AssertionError):
            Similarity(df_features=pd_df_simple_table)

    def test_check_nulls_in_feature_columns(self):

        df_nulls_features = self.spark.read.csv('tests/fixtures/similarity/nulls_features.csv', header=True)
        df_no_nulls_features = self.spark.read.csv('tests/fixtures/similarity/no_nulls_features.csv', header=True)

        Similarity(df_features=df_no_nulls_features)

        with self.assertRaises(AssertionError):
            Similarity(df_features=df_nulls_features)

    def test_check_is_numerical_data(self):

        df_numerical = self.spark.read.csv('tests/fixtures/similarity/numerical_data.csv', header=True)

        columns_to_convert = [col for col in df_numerical.columns if 'id' not in col]
        df_numerical_int = df_numerical
        df_numerical_float = df_numerical

        for col in columns_to_convert:
            df_numerical_int = df_numerical_int.withColumn(col, f.col(col).cast(IntegerType()))
            df_numerical_float = df_numerical_float.withColumn(col, f.col(col).cast(DoubleType()))

        Similarity(df_features=df_numerical_int)
        Similarity(df_features=df_numerical_float)

        with self.assertRaises(AssertionError):
            Similarity(df_features=df_numerical)


