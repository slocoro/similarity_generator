from spark_utils import create_spark_session

from preprocess import Preprocess
from similarity import Similarity
from utils import create_timestamp
from utils import create_parameters_table

import os
import sys


spark = create_spark_session('generate_similarities')

file_name = sys.argv[1]

df_recipe_info = spark.read.csv(f'data/{file_name}', header=True)

COLUMNS = 'all'
INDEX_COLUMN = 'recipe_id'
SIMILARITY_TYPE = 'euclidean'

etl_created = create_timestamp()

preprocessor = Preprocess(df_labels=df_recipe_info,
                          columns='all')
df_recipe_features = preprocessor.preprocess()
pd_df_recipe_features = df_recipe_features.toPandas()
features_dir = f'output/{etl_created}/features'
os.makedirs(features_dir)
pd_df_recipe_features.to_csv(f'{features_dir}/features.csv', index=False)


similarity = Similarity(df_features=df_recipe_features,
                        index_column=INDEX_COLUMN,
                        similarity_type=SIMILARITY_TYPE)
similarities = similarity.generate()
pd_df_similarities_wide = similarities[0]
pd_df_similarities_long = similarities[1]

similarities_dir = f'output/{etl_created}/similarities'
os.makedirs(similarities_dir)
pd_df_similarities_wide.to_csv(f'{similarities_dir}/similarities_wide.csv', index=True)
pd_df_similarities_long.to_csv(f'{similarities_dir}/similarities_long.csv', index=False)

parameters_dir = f'output/{etl_created}/parameters'
os.makedirs(parameters_dir)
pd_df_parameters = create_parameters_table(similarity_type=SIMILARITY_TYPE,
                                           index_column=INDEX_COLUMN,
                                           columns=COLUMNS)
pd_df_parameters.to_csv(f'{parameters_dir}/parameters.csv', index=False)


spark.stop()

