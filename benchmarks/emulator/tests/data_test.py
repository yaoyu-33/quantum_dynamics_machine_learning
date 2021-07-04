"""Test emulator."""
import mock
import unittest
import numpy

import emulator.data


train_data = numpy.array([
    [0, 1, 255],
    [0, 10, 255],
])

train_labels = numpy.array([0, 1])

valid_data = numpy.array([
    [0, 8, 255],
    [0, 1, 255],
])

valid_labels = numpy.array([1, 0])


class EmulatorDataTest(unittest.TestCase):
    """Test datasets."""

    def setUp(self) -> None:
        """Load before every test."""
        pass

    @mock.patch(
        'tensorflow.keras.datasets.mnist.load_data',
        mock.Mock(return_value=((train_data, train_labels), (None, None))))
    def test_training_data(self):
        """Read the training data."""
        x, y = emulator.data.get_train_data('path')

        r = numpy.array([[0., 1/255., 1.], [0., 10/255., 1.]], dtype='float32')
        numpy.testing.assert_array_equal(x, r)

        numpy.testing.assert_array_equal(y, numpy.array([0, 1], dtype='int32'))

    @mock.patch(
        'tensorflow.keras.datasets.mnist.load_data',
        mock.Mock(return_value=((None, None), (valid_data, valid_labels))))
    def test_validation_data(self):
        """Read the training data."""
        x, y = emulator.data.get_validation_data('path')

        r = numpy.array([[0., 8/255., 1.], [0., 1/255., 1.]], dtype='float32')
        numpy.testing.assert_array_equal(x, r)

        numpy.testing.assert_array_equal(y, numpy.array([1, 0], dtype='int32'))

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
