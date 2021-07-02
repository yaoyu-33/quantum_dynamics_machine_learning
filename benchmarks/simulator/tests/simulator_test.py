"""Test simulator."""
import mock
import unittest
import numpy
import tensorflow

import simulator.simulator


simulation_file = '\n'.join([
    'timestamp: 0.00000',
    'params: 12 100.000000 0.0005 36.0 2.6 6.8 4.1 7.0 0.0',
    'psi_re: 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0 0.0 0.0 0.0',
    'psi_im: 0.0 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0 0.0 0.0',
    'pot: 0.0 0.0 0.0 0.0 0.3 0.3 0.3 0.0 0.0 0.0 0.0 0.0',
    '',
    'timestamp: 0.10000',
    'params: 13 100.000000 0.0005 36.0 2.6 6.8 4.1 7.0 0.0',
    'psi_re: 0.0 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0 0.0 0.0',
    'psi_im: 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0 0.0',
    'pot: 0.0 0.0 0.0 0.0 0.3 0.3 0.3 0.0 0.0 0.0 0.0 0.0',
    '',
    'timestamp: 0.20000',
    'params: 14 100.000000 0.0005 36.0 2.6 6.8 4.1 7.0 0.0',
    'psi_re: 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0 0.0',
    'psi_im: 0.0 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.3 0.2 0.1 0.0',
    'pot: 0.0 0.0 0.0 0.0 0.3 0.3 0.3 0.0 0.0 0.0 0.0 0.0',
])


class SimulatorTest(unittest.TestCase):
    """Test input."""

    def setUp(self) -> None:
        """Load before every test."""
        pass

    @mock.patch(
        'tensorflow.io.gfile.GFile',
        new_callable=mock.mock_open, read_data=simulation_file)
    def test_data_reading(self, mocked_file):
        """Test the function reading the temporary simulation data."""
        data = simulator.simulator.retrieve_data('file_name', pot_scalar=10)

        mocked_file.assert_called_with("file_name")

        self.assertSetEqual(
            set(data.keys()),
            {'timestamp', 'params', 'psi_re', 'psi_im', 'pot'})

        numpy.testing.assert_array_equal(
            data['timestamp'],
            numpy.array([0., 0.1, 0.2]))

        numpy.testing.assert_array_equal(
            data['params'],
            numpy.array([
                [12., 100.0, 0.0005, 36.0, 2.6, 6.8, 4.1, 7.0, 0.0],
                [13., 100.0, 0.0005, 36.0, 2.6, 6.8, 4.1, 7.0, 0.0],
                [14., 100.0, 0.0005, 36.0, 2.6, 6.8, 4.1, 7.0, 0.0],
            ]))

        numpy.testing.assert_array_equal(
            data['psi_re'],
            numpy.array([
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0],
            ]))

        numpy.testing.assert_array_equal(
            data['psi_im'],
            numpy.array([
                [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0],
            ]))

        numpy.testing.assert_array_equal(
            data['pot'],
            numpy.array([
                [0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]))

    def test_train_features(self):
        """Test conversion to train.features."""
        x = [0.0, 0.1, 0.2, 0.3]
        self.assertEqual(
            type(simulator.simulator.create_float_feature(x)),
            tensorflow.train.Feature)

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
