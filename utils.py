# this file will include utility functions for loading the dataset, plotting, etc.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
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

    multi_labels = []

    for cat_label in categorical_labels:
        new_label = np.zeros(18, dtype=int)
        type1, type2 = cat_label
        new_label[type1-2] = 1
        if type2 >= 2:
            new_label[type2-2] = 1

        multi_labels.append(new_label)

    label_batch = np.array(multi_labels)

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

    # Convert RGBA images to RGB
    if image_batch.shape[-1] == 4:
        image_batch = compose_alpha(image_batch*255)

    dataset = tf.data.Dataset.from_tensor_slices((image_batch, label_batch))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(dataset))

    # Calculate the number of examples for the training set
    train_size = int(0.8 * len(dataset))  # 80% for training

    # Split the dataset
    train_dataset = dataset.take(train_size)  # Take first 80%
    val_dataset = dataset.skip(train_size)    # Skip the first 80%, take the rest

    # Batch and prefetch
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

def plot_history(histories):
    # Create a figure with two subplots: one for loss, one for accuracy
    fig, axs = plt.subplots(len(histories), 2, figsize=(11, len(histories)*3))
    if len(histories) == 1:
        axs = [axs]
    for i, (history, (ax1, ax2)) in enumerate(zip(histories,axs)):
        # Iterate over each history and label to plot on the same figure
        # Plot all metrics included in the history object
        for key in history.history.keys():
            # check if the key includes "val" or not
            if 'val' in key:
                ax2.plot(history.history[key], label='Model '+str(i+1)+' '+key)
            else:
                ax1.plot(history.history[key], label='Model '+str(i+1)+' '+key)

        # Customize loss subplot
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss/Accuracy')
        ax1.set_title('Training Performance over Epochs')
        ax1.legend()
        ax1.set_xlim(left=0)  # Ensure the x-axis starts at 0

        # Customize accuracy subplot
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss/Accuracy')
        ax2.set_title('Validation Performance over Epochs')
        ax2.legend()
        ax2.set_xlim(left=0)  # Ensure the x-axis starts at 0
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def create_types_wildcard():
    pokemon_types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
    type_combinations = []
    for type in pokemon_types:
        for type2 in pokemon_types:
            if type == type2:
                type_combinations.append(type)
            else:
                type_combinations.append(type + ' ' + type2)

    with open('pokemon_type_combinations.txt', 'w') as f:
        for item in type_combinations:
            f.write("%s\n" % item)

def create_csv_synthetic(dataset_name = 'synthetic_images'):
    filelist = os.listdir('images/'+dataset_name)
    name = []
    type1 = []
    type2 = []
    for file in filelist:
        name.append(file.split('.')[0])
        type1.append(file.split('_')[0].capitalize())

        if (file.split('_')[1].isalpha()):
            type2.append(file.split('_')[1].capitalize())
        else:
            type2.append(None)

    poke_dict = {'Name': name, 'Type1': type1, 'Type2': type2, 'Evolution': None}
    df = pd.DataFrame(poke_dict)
    df.to_csv(dataset_name+'.csv', index=False)

def data_exploration(csv):

    df = pd.read_csv(csv)
    df = df.drop('Evolution', axis = 1)
    df = df.sort_values(by='Name')
    df = df.fillna('None')
    print(df.head())

    type1_counts = df['Type1'].value_counts()
    type2_counts = df['Type2'].value_counts()
    plt.figure(figsize=(12, 6))
    #type1_counts += type2_counts

    plt.subplot(1, 2, 1)
    type1_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Type1')
    plt.xlabel('Type1')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    type2_counts.plot(kind='bar', color='lightgreen')
    plt.title('Distribution of Type2')
    plt.xlabel('Type2')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def compose_alpha(image_with_alpha):
    image_with_alpha = image_with_alpha.astype(np.float32)
    image, alpha = image_with_alpha[..., :3], image_with_alpha[..., 3:] / 255.0
    image = image * alpha + (1.0 - alpha) * 255.0
    image = image.astype(np.uint8)
    return image

if __name__ == '__main__':
    #create_csv_synthetic(dataset_name='synthetic_pokemon_v2')
    csv = "pokemon_image_dataset.csv"
    csv2 = "synthetic_pokemon.csv"
    csv3 = "synthetic_pokemon_v2.csv"
    #data_exploration(csv)
    pass