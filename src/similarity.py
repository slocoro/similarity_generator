from pyspark.sql import DataFrame
import pyspark.sql.functions as f

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd


class Similarity(object):
    """
    Class to generate similarity scores.

    """

    def __init__(self, df_features, index_column='recipe_id', similarity_type='cosine'):
        """

        :param df_features: spark data frame, contains labels in first column and int features in remaining

        """

        self.df_features = df_features
        self.index_column = index_column
        self.similarity_type = similarity_type

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

    def generate(self):
        """
        Generates similarity scores.

        :return: pandas data frame (wide), numpy ndarray (wide), pandas data frame (long)
        """

        pd_df_similarity = self.df_features.toPandas()
        similarity_indexes = pd_df_similarity[self.index_column].tolist()
        pd_df_similarity_no_index = pd_df_similarity.drop(columns=[self.index_column])

        if self.similarity_type == 'cosine':
            mat_similarity = cosine_similarity(pd_df_similarity_no_index)
        elif self.similarity_type == 'euclidean':
            mat_similarity = euclidean_distances(pd_df_similarity_no_index)
        else:
            raise ValueError('Unknown "similarity_type".')

        pd_df_similarity = pd.DataFrame(mat_similarity, index=similarity_indexes, columns=similarity_indexes)

        pd_df_similarity_long = self.convert_to_long_format(pd_df_similarity)

        return pd_df_similarity, mat_similarity, pd_df_similarity_long

    @staticmethod
    def convert_to_long_format(pd_df_similarity):
        """
        Converts wide similarities to long.

        :param pd_df_similarity: pandas data frame
        :return:
        """

        pd_df_similarity_copy = pd_df_similarity.copy(deep=True)

        pd_df_similarity_copy.insert(loc=0, column='recipe_id', value=pd_df_similarity_copy.index)

        pd_df_similarity_long = pd.melt(pd_df_similarity_copy,
                                        id_vars=['recipe_id'],
                                        value_vars=pd_df_similarity_copy.columns.values.tolist()[1:],
                                        var_name='recipe_id_2',
                                        value_name='similarity')

        pd_df_similarity_long.rename(columns={'recipe_id': 'recipe_id_1'}, inplace=True)

        return pd_df_similarity_long
