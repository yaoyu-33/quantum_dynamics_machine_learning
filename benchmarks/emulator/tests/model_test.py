"""Test emulator."""
import unittest

import emulator.model


class EmulatorModelTest(unittest.TestCase):
    """Test datasets."""

    def setUp(self) -> None:
        """Load before every test."""
        pass

    def test_input_shape(self):
        """Test the input shape."""
        model = emulator.model.build()
        self.assertEqual(list(model.input.shape), [None, 28, 28])

    def test_output_shape(self):
        """Test the output shape."""
        model = emulator.model.build()
        self.assertEqual(list(model.output.shape), [None, 10])

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
