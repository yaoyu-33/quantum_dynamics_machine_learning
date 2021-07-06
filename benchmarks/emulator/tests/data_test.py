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


def return_file(*args):
    if args[0] == 'train_data':
        return train_data
    elif args[0] == 'train_labels':
        return train_labels
    elif args[0] == 'valid_data':
        return valid_data
    elif args[0] == 'valid_labels':
        return valid_labels


class EmulatorDataTest(unittest.TestCase):
    """Test datasets."""

    def setUp(self) -> None:
        """Load before every test."""
        pass

    @mock.patch('numpy.load', mock.Mock(side_effect=return_file))
    def test_training_data(self):
        """Read the training data."""
        x, y = emulator.data.get_train_data(
            data_path='train_data',
            labels_path='train_labels'
        )

        r = numpy.array([[0., 1/255., 1.], [0., 10/255., 1.]], dtype='float32')
        numpy.testing.assert_array_equal(x, r)

        numpy.testing.assert_array_equal(y, numpy.array([0, 1], dtype='int32'))

    @mock.patch('numpy.load', mock.Mock(side_effect=return_file))
    def test_validation_data(self):
        """Read the training data."""
        x, y = emulator.data.get_validation_data(
            data_path='valid_data',
            labels_path='valid_labels'
        )

        r = numpy.array([[0., 8/255., 1.], [0., 1/255., 1.]], dtype='float32')
        numpy.testing.assert_array_equal(x, r)

        numpy.testing.assert_array_equal(y, numpy.array([1, 0], dtype='int32'))

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
