"""Test emulator."""
import unittest
import numpy
import tensorflow

import emulator.training


train_data = numpy.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

train_labels = numpy.array([0, 1, 0, 1])


class EmulatorTrainingTest(unittest.TestCase):
    """Test datasets."""

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

    def test_training(self):
        """Train the model."""
        emulator.training.train(
            self.model, train_data, train_labels, epochs=25, batch_size=4)

        print('***', self.model.predict([[0, 1]]))
        print('***', self.model.predict([[1, 1]]))
        print('***', self.model.predict([[0, 0]]))
        print('***', self.model.predict([[1, 0]]))

        self.assertLess(self.model.predict([[0, 1]])[0][0], 0.5)
        self.assertGreater(self.model.predict([[1, 1]])[0][1], 0.5)

        self.assertGreater(self.model.predict([[0, 0]])[0][0], 0.5)
        self.assertLess(self.model.predict([[1, 0]])[0][1], 0.5)

    def tearDown(self) -> None:
        """Load after all tests are executed."""
        pass


if __name__ == '__main__':
    """Run unittests."""
    unittest.main()
