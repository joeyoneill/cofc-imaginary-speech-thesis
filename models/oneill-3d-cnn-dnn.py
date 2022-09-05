# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##################################################################
# 3D-CNN-DNN Model build                                         #
##################################################################

# Input
a1_inputs = tf.keras.Input(shape = (125, 10, 11, 1))

# 3-Layer 3D CNN
a1_model = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(a1_inputs)
a1_model = layers.Conv3D(64, (3, 3, 3), activation = 'relu')(a1_model)
a1_model = layers.Conv3D(128, (5, 4, 3), activation = 'relu')(a1_model)

# Flatten to Dense
a1_model = layers.Flatten()(a1_model)

# FC Layers
a1_model = layers.Dense(1024, activation='relu')(a1_model)
a1_model = layers.Dense(1024, activation='relu')(a1_model)
a1_model = layers.Dense(1024, activation='relu')(a1_model)

# Softmax Output
a1_outputs = layers.Dense(5, activation='softmax')(a1_model)

# Model Creation
a_3d_cnn = keras.Model(inputs=a1_inputs, outputs=a1_outputs)
