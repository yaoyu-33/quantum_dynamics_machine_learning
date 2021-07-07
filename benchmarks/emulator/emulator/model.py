"""Define the neural network architecture."""
import tensorflow as tf
from tensorflow.keras import layers

class RNNModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.flatten = layers.Reshape((config.input_frames, config.window_size * config.input_channels)) 
        self.dense1 = layers.Dense(config.hidden_size, activation="relu")
        self.gru = tf.keras.layers.GRU(config.hidden_size, return_state=True, activation='relu', dropout=config.dropout_rate)
        self.dense2 = layers.Dense(config.window_size * config.output_channels)
        self.reshape = layers.Reshape((config.window_size, config.output_channels))

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        output, final_state = self.gru(x, initial_state=None, training=training)
        x = tf.concat((output, final_state), axis=-1)
        x = self.dense2(x)
        x = self.reshape(x)
        return x

# conf_gru = Config(model_name="demo-gru")
# m_gru = RNNModel(conf_gru)
# m_gru.build((None, conf_gru.input_frames, conf_gru.window_size, conf_gru.input_channels))
# m_gru.summary()

# def build(hidden, dropout, lr, momentum):
#     """Build Neural Network.

#     Args:
#         hidden (int): number of hidden neurons
#         dropout (float): dropout ratio
#         lr (float): learning ratio for the optimizer
#         momentum (float): momentum parameter for the SGD optimizer

#     Returns:
#         (tensoflow.keras.Model): compiled neural network.
#     """
#     # TODO: Define here our recurrent neural network
#     # TODO: Update the arguments (parameters that we want to optimize by ray)

#     model = tensorflow.keras.models.Sequential([
#         tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
#         tensorflow.keras.layers.Dense(hidden, activation="relu"),
#         tensorflow.keras.layers.Dropout(dropout),
#         tensorflow.keras.layers.Dense(10, activation="softmax")
#     ])

#     model.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=tensorflow.keras.optimizers.SGD(
#             lr=lr, momentum=momentum),
#         metrics=["accuracy"])

#     return model
