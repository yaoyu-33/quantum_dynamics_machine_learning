"""Test emulator."""
import unittest
import tensorflow as tf
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
class EmulatorModelTest(unittest.TestCase):
    """Test datasets."""

    def setUp(self) -> None:
        """Load before every test."""
        pass

    def test_output_shape(self):
        """Test the output shape."""
        model = emulator.model.RNNModel(config)
        inp = tf.random.uniform([config.train_batch, config.input_frames, config.window_size, config.input_channels])
        res = model(inp)
        self.assertEqual(res.numpy().shape, (config.train_batch,config.window_size,config.output_channels))

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
