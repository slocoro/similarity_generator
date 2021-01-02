import unittest

from pyspark.sql import SparkSession


class PySparkTestCase(unittest.TestCase):
    """
    Class to test modules that use pyspark.

    """

    def setUp(self):

        self.spark = SparkSession \
            .builder \
            .appName("unit_testing") \
            .config("spark.executor.memory", "30g") \
            .enableHiveSupport() \
            .getOrCreate()

    def tearDown(self):

        self.spark.stop()
