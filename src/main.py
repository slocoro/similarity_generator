from spark_utils import create_spark_session

from preprocess import Preprocessor


spark = create_spark_session('generate_similarities')

df_recipe_info = spark.read.csv('data/recipes_info.csv', header=True)
columns = 'all'

preprocessor = Preprocessor(df_recipe_info=df_recipe_info,
                            columns='all')

df_recipe_features = preprocessor.preprocess()

df_recipe_features.show()

print(df_recipe_features.dtypes)