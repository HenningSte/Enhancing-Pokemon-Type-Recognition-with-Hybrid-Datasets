import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from utils import load


csv = "pokemon_image_dataset.csv"
image_path = "./images"

dataset = load(csv, image_path)

# dataset.save('pokemon_image_dataset')

# dataset = tf.data.Dataset.load('pokemon_image_dataset')

model = models.Sequential([
    # Input layer (not explicitly needed in Sequential, but shows input shape)
    layers.InputLayer(shape=(120, 120, 4)),
    
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
    layers.Dense(20, activation='softmax', name='type1'), 
    layers.Dense(20, activation='softmax', name='type2')
])

# Compile the model
model.compile(optimizer='adam',
              loss={
                  'type1':'sparse_categorical_crossentropy',
                  'type2':'sparse_categorical_crossentropy'
                },
              metrics={
                  'type1':'accuracy',
                  'type2':'accuracy'
              })

history = model.fit(
    dataset,
    epochs = 10,
    verbose = 1
)

print("Training History:", history.history)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.show()