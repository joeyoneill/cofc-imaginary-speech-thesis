# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##################################################################
# ZHANG et al.'s PARALLEL CRNN                                   #
##################################################################

# 3-Layer 2D CNN

# Input: Spatial Matrices
inputs_1 = tf.keras.Input(shape = (125, 10, 11, 1))

# 2D CNN Layers
model_1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs_1)
model_1 = layers.Conv2D(64, (3, 3), activation='relu')(model_1)
model_1 = layers.Conv2D(128, (3, 2), activation='relu')(model_1)

# Flatten CNN
model_1 = layers.Flatten()(model_1)

# CNN FC Layer
model_1 = layers.Dense(1024, activation='relu')(model_1)

##################################################################
# 2-Layer LSTM RNN                                               #
##################################################################

# Input: Voltage-at-time array
inputs_2 = tf.keras.Input(shape = (125, 16))

# LSTM Layer 1
model_2 = layers.LSTM(16, return_sequences = True)(inputs_2)

# LSTM Layer 2
model_2 = layers.LSTM(16)(model_2)

# LSTM FC Layer
model_2 = layers.Dense(1024)(model_2)

##################################################################
# Model Concatenation and output
##################################################################

# concat the RNN and the CNN
model_concat = tf.concat([model_1, model_2], axis=1)

# SoftMax Output
outputs = layers.Dense(2, activation='softmax')(model_concat)

# Model Creation
parallel_crnn = keras.Model(inputs=[inputs_1, inputs_2], outputs=[outputs])
