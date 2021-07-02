"""Test simulator."""
import unittest

import simulator.simulator


class SimulatorTest(unittest.TestCase):
    """Test input."""

    def setUp(self):
        """Load before every test."""

    def test_addition(self):
        """Test auxiliary function."""
        self.assertEqual(simulator.simulator.func(4), 6)


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
