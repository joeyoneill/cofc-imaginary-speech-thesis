# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##################################################################
# ZHANG et al.'s Cascade CRNN                                    #
##################################################################

# Input
crnn_inputs = tf.keras.Input(shape = (125, 10, 11, 1))

# 3-Layer 2D CNN
x = layers.Conv2D(32, (3, 3), activation='relu')(crnn_inputs)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)

# Flatten ConvNet Output
x = layers.Flatten()(x)

# Fully Connected layer
x = layers.Dense(1024, activation='relu')(x)

# Reshape Feature Vector to fit in LSTM cell
x = layers.Reshape((1, x.shape[1]))(x)

# 2-Layer LSTM RNN
# Changing units to 1024 from 64, 16 (original) respectively
x = layers.LSTM(1024, return_sequences = True)(x)
x = layers.LSTM(1024)(x)

# Output
crnn_outputs = layers.Dense(2, activation='softmax')(x)

# Model Creation
cascade_crnn = keras.Model(inputs=crnn_inputs, outputs=crnn_outputs)
