# this file will include utility functions for loading the dataset, plotting, etc.

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import to_categorical

def load(csv, image_path):

    df = pd.read_csv(csv)
    df = df.drop('Evolution', axis = 1)
    df = df.sort_values(by='Name')
    df = df.fillna('None')
    labels = df[['Type1', 'Type2']].values

    string_lookup = tf.keras.layers.StringLookup()

    string_lookup.adapt([item for sublist in labels for item in sublist])

    categorical_labels = np.array([string_lookup(label) for label in labels])

    type1, type2 = categorical_labels[:,0], categorical_labels[:,1]

    print("Vocabulary:", string_lookup.get_vocabulary())

    image_arrays = []
    # Loop over all files in the image folder
    for filename in os.listdir(image_path):
        if filename.endswith('.png'):  # Check if the file is a PNG
            # Construct the full file path
            file_path = os.path.join(image_path, filename)
            
            # Load the image as a NumPy array
            image_array = plt.imread(file_path)
            
            # Append the image array to the list
            image_arrays.append(image_array)

    # Convert the list of image arrays to a NumPy array (if all images have the same shape)
    image_batch = np.array(image_arrays)

    dataset = tf.data.Dataset.from_tensor_slices((image_batch, {'type1':type1, 'type2':type2}))

    dataset = dataset.shuffle(buffer_size=len(image_batch)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
