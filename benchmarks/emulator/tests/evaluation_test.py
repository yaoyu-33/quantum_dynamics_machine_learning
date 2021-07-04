"""Test emulator."""
import unittest
import numpy
import tensorflow

import emulator.evaluation


data = numpy.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

labels = numpy.array([0, 1, 0, 1])


class EmulatorEvaluateTest(unittest.TestCase):
    """Test evaluation."""

    def setUp(self) -> None:
        """Load before every test."""
        self.model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Flatten(input_shape=(2,)),
            tensorflow.keras.layers.Dense(2, activation="softmax")
        ])

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tensorflow.keras.optimizers.SGD(lr=0.1, momentum=0.9),
            metrics=["accuracy"])

    def test_evaluation(self):
        """Test the model evaluation."""
        score = emulator.evaluation.evaluate(self.model, data, labels)

        print('***', score)

        self.assertEqual(type(score), float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
