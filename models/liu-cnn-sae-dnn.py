# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##################################################################
# Liu et al/'s CNN-SAE-DNN                                       #
##################################################################
# Input
inputs = tf.keras.Input(shape = (125, 10, 11))

# 2D CNN
model = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
model = layers.Conv2D(64, (3, 3), activation='relu')(model)

# Flatten into SAE
model = layers.Flatten()(model)

# SAE
model = layers.Dense(5120, activation='relu')(model)
model = layers.Dense(120, activation='relu')(model)
model = layers.Dense(5120, activation='relu')(model)

# DNN
model = layers.Dense(1024, activation='relu')(model)
model = layers.Dense(1024, activation='relu')(model)
model = layers.Dense(1024, activation='relu')(model)

# Softmax Output
outputs = layers.Dense(2, activation='softmax')(model)

# Model Creation
liu_model = keras.Model(inputs=inputs, outputs=outputs)
