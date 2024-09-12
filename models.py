import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def custom_conv_net(input_shape):
    model = models.Sequential([
        # Input layer (not explicitly needed in Sequential, but shows input shape)
        layers.InputLayer(shape=input_shape),
        
        # Add some convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the output
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),

        # Two output layers for both types
        layers.Dense(18, activation='sigmoid'), 
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss= 'binary_crossentropy',
        metrics=['accuracy']
        )
    
    return model