from pyspark.sql import SparkSession


def create_spark_session(name):
    """
    Creates spark session.

    :param name: string
    :return: None
    """
    spark = SparkSession\
        .builder\
        .appName(name)\
        .config('spark.executor.memory', '30g')\
        .enableHiveSupport()\
        .getOrCreate()
    return spark

