import unittest

import tensorflow as tf

from optimizer_network import LSTMNetworkPerParameter


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = LSTMNetworkPerParameter()
        return super().setUp()

if __name__ == "__main__":
    unittest.main()
