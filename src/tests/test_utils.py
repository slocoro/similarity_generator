import unittest

from utils import create_timestamp


class TestUtils(unittest.TestCase):

    def test_create_timestamp(self):

        timestamp = create_timestamp()

        self.assertTrue('_' in timestamp)
        self.assertTrue(len(timestamp), 4+2+2+1+4)


