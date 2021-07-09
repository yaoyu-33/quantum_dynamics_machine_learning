"""Test emulator."""
import unittest
import numpy
import tensorflow as tf
import emulator.evaluation
import emulator.model


class Config:
    """Task config."""
    def __init__(self):
        """Configuration Settings for Training and Models."""        
        # Training Settings
        self.input_frames = 4
        self.input_channels = 3
        self.output_channels = 2
        self.window_size = 23
        self.dropout_rate = 0.
        self.hidden_size = 690
        self.train_batch = 128

config = Config()
data = {'psi_re':numpy.random.rand(20,20),'psi_im':numpy.random.rand(20,20),'pot':numpy.random.rand(20,20)}
result = tf.random.uniform([10,20,2])

class EmulatorEvaluateTest(unittest.TestCase):
    """Test evaluation."""

    def setUp(self) -> None:
        """Load before every test."""
        self.model = emulator.model.RNNModel(config)
        

    def test_avg_corr(self):
        """Test the average correlation function."""
        score = emulator.evaluation.get_avg_corr(config, data, result, frames=10)

        print('***', score)

        self.assertEqual(type(score), numpy.float64)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_get_result(self):
        """Test get_result function."""
        res = emulator.evaluation.get_result(self.model, config, data, frames=10)
        self.assertEqual(type(res), tf.python.framework.ops.EagerTensor)
        self.assertEqual(res.shape,(10,20,config.output_channels))

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
