import unittest

from utils import create_timestamp
from utils import create_parameters_table


class TestUtils(unittest.TestCase):

    def test_create_timestamp(self):

        timestamp = create_timestamp()

        self.assertTrue('_' in timestamp)
        self.assertTrue(len(timestamp), 4+2+2+1+4)

    def test_create_parameters_table(self):

        columns_all = 'all'
        columns_sub = ['country', 'protein']
        similarity_type = 'cosine'
        index_column = 'recipe_id'

        pd_df_all = create_parameters_table(similarity_type=similarity_type,
                                            columns=columns_all,
                                            index_column=index_column)

        self.assertEqual(pd_df_all.shape[0], 1)
        self.assertEqual(pd_df_all.shape[1], 3)

        pd_df_sub = create_parameters_table(similarity_type=similarity_type,
                                            columns=columns_sub,
                                            index_column=index_column)

        self.assertEqual(pd_df_sub.shape[0], 1)
        self.assertEqual(pd_df_sub.shape[1], 3)

        columns_sub_check = list(pd_df_sub['columns'].values)[0]
        self.assertEqual(columns_sub_check, 'country, protein')

