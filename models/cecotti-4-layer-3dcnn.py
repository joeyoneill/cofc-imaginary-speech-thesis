# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##################################################################
# (MODIFIED) Cecotti and Jha's 4-Layer 3DCNN                     #
##################################################################

# Input
c4_inputs = tf.keras.Input(shape = (125, 10, 11, 1))

# 4-Layer 3D CNN
c4_model = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(c4_inputs)
c4_model = layers.Conv3D(64, (3, 3, 3), activation = 'relu')(c4_model)
c4_model = layers.Conv3D(128, (3, 3, 3), activation = 'relu')(c4_model)
c4_model = layers.Conv3D(256, (3, 2, 3), activation = 'relu')(c4_model)

# Flatten to Dense
c4_model = layers.Flatten()(c4_model)

# FC Layers (DNN)
c4_model = layers.Dense(1024, activation='relu')(c4_model)
c4_model = layers.Dense(1024, activation='relu')(c4_model)
c4_model = layers.Dense(1024, activation='relu')(c4_model)

# SoftMax Output
c4_outputs = layers.Dense(2, activation='softmax')(c4_model)

# Model Creation
c4_3d_cnn = keras.Model(inputs=c4_inputs, outputs=c4_outputs)
