"""Define the neural network architecture."""
import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# The original plain model
class RNNModel(tf.keras.Model):
    """Machine Learning Based Emulator (RNN Type)."""

    def __init__(self, config, reg=False, type=None):
        """Initialization."""
        super().__init__()
        self.flatten = layers.Reshape(
            (config.input_frames, config.window_size*config.input_channels))
        if reg:
            self.dense1 = layers.Dense(
            config.hidden_size, activation="relu", kernel_regularizer = type) #It has Hidden Size (69) units
        else:
            self.dense1 = layers.Dense(
                config.hidden_size, activation="relu") #It has Hidden Size (69) units
        self.gru = tf.keras.layers.GRU(
            config.hidden_size, return_state=True, activation='relu',
            dropout=config.dropout_rate)
        self.dense2 = layers.Dense(
            config.window_size * config.output_channels)
        self.reshape = layers.Reshape(
            (config.window_size, config.output_channels))

    def call(self, inputs, training=False, mask=None):
        """Call the model on new inputs.""" 
        #INPUT SHAPE: [128,4,23,3] --> [batch, timesteps, window size, channels]

        x = self.flatten(inputs) #FLATTEN: [128, 4, 69] --> [batch, timesteps, window size * channels]

        x = self.dense1(x) #DENSE: [128, 4, 69] --> [batch, timesteps, hidden_size]

        output, final_state = self.gru(
            x, initial_state=None, training=training) #OUTPUT and FINAL_STATE: [128,69] --> [batch, hidden_size]

        x = tf.concat((output, final_state), axis=-1) #CONCAT: [128,138] --> [batch, 69+69]

        x = self.dense2(x) #DENSE2: [128, 46] --> [batch, window_size*output_channels]

        x = self.reshape(x) #RESHAPE: [128, 23, 2] --> [batch, window_size, output_channels]
        return x







#EDITS

# The model with no dense layer
class RNNModel_nodense(tf.keras.Model):
    """Machine Learning Based Emulator (RNN Type)."""

    def __init__(self, config):
        """Initialization."""
        super().__init__()
        self.flatten = layers.Reshape(
            (config.input_frames, config.window_size*config.input_channels))
        self.dense1 = layers.Dense(
            config.hidden_size, activation="relu") #It has Hidden Size (69) units
        self.gru = tf.keras.layers.GRU(
            config.hidden_size, return_state=True, activation='relu',
            dropout=config.dropout_rate)
        self.dense2 = layers.Dense(
            config.window_size * config.output_channels)
        self.reshape = layers.Reshape(
            (config.window_size, config.output_channels))

    def call(self, inputs, training=False, mask=None):
        """Call the model on new inputs.""" 
        #INPUT SHAPE: [128,4,23,3] --> [batch, timesteps, window size, channels]

        x = self.flatten(inputs) #FLATTEN: [128, 4, 69] --> [batch, timesteps, window size * channels]

        # x = self.dense1(x) #DENSE: [128, 4, 69] --> [batch, timesteps, hidden_size]

        output, final_state = self.gru(
            x, initial_state=None, training=training) #OUTPUT and FINAL_STATE: [128,69] --> [batch, hidden_size]

        x = tf.concat((output, final_state), axis=-1) #CONCAT: [128,138] --> [batch, 69+69]

        x = self.dense2(x) #DENSE2: [128, 46] --> [batch, window_size*output_channels]

        x = self.reshape(x) #RESHAPE: [128, 23, 2] --> [batch, window_size, output_channels]
        return x


# The model with convolutional layer
class RNNModel_conv(tf.keras.Model):
    """Machine Learning Based Emulator (RNN Type)."""

    def __init__(self, config):
        """Initialization."""
        super().__init__()
        self.flatten = layers.Reshape(
            (config.input_frames, config.window_size*config.input_channels))
        self.dense1 = layers.Dense(
            config.hidden_size, activation="relu") #It has Hidden Size (69) units
        
        self.conv = layers.Conv2D(3,(1,1),activation='relu')
        self.gru = tf.keras.layers.GRU(
            config.hidden_size, return_state=True, activation='relu',
            dropout=config.dropout_rate)
        self.dense2 = layers.Dense(
            config.window_size * config.output_channels)
        self.reshape = layers.Reshape(
            (config.window_size, config.output_channels))

    def call(self, inputs, training=False, mask=None):
        """Call the model on new inputs.""" 
        #INPUT SHAPE: [128,4,23,3] --> [batch, timesteps, window size, channels]

        # x = self.flatten(inputs) #FLATTEN: [128, 4, 69] --> [batch, timesteps, window size * channels]

        # x = self.dense1(x) #DENSE: [128, 4, 69] --> [batch, timesteps, hidden_size]

        x = self.conv(inputs)

        x = self.flatten(x)

        output, final_state = self.gru(
            x, initial_state=None, training=training) #OUTPUT and FINAL_STATE: [128,69] --> [batch, hidden_size]

        x = tf.concat((output, final_state), axis=-1) #CONCAT: [128,138] --> [batch, 69+69]

        x = self.dense2(x) #DENSE2: [128, 46] --> [batch, window_size*output_channels]

        x = self.reshape(x) #RESHAPE: [128, 23, 2] --> [batch, window_size, output_channels]
        return x

# The model with three dense layers for each channel
class RNNModel_3dense(tf.keras.Model):
    """Machine Learning Based Emulator (RNN Type)."""

    def __init__(self, config):
        """Initialization."""
        super().__init__()
        self.flatten = layers.Reshape(
            (config.input_frames, config.window_size*config.input_channels))
        self.dense1 = layers.Dense(
            config.hidden_size, activation="relu") #It has Hidden Size (69) units
        self.gru = tf.keras.layers.GRU(
            config.hidden_size, return_state=True, activation='relu',
            dropout=config.dropout_rate)
        self.dense2 = layers.Dense(
            config.window_size * config.output_channels)
        self.reshape = layers.Reshape(
            (config.window_size, config.output_channels))
        
        self.dense_23 = layers.Dense(23, activation='relu')

    def call(self, inputs, training=False, mask=None):
        """Call the model on new inputs.""" 
        #INPUT SHAPE: [128,4,23,3] --> [batch, timesteps, window size, channels]

        x1 = inputs[:,:,:,0]
        x2 = inputs[:,:,:,1]
        x3 = inputs[:,:,:,2]

        x1 = self.dense_23(x1)
        x2 = self.dense_23(x2)
        x3 = self.dense_23(x3)
        
        x = tf.concat((x1,x2,x3), axis=-1)
        
        output, final_state = self.gru(
            x, initial_state=None, training=training) #OUTPUT and FINAL_STATE: [128,69] --> [batch, hidden_size]

        x = tf.concat((output, final_state), axis=-1) #CONCAT: [128,138] --> [batch, 69+69]

        x = self.dense2(x) #DENSE2: [128, 46] --> [batch, window_size*output_channels]

        x = self.reshape(x) #RESHAPE: [128, 23, 2] --> [batch, window_size, output_channels]
        return x

