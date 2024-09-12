# this file will include utility functions for loading the dataset, plotting, etc.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load(csv, image_path):
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

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

def create_csv_synthetic():
    filelist = os.listdir('synthetic_images')
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
    df.to_csv('synthetic_pokemon.csv', index=False)

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

if __name__ == '__main__':
    create_csv_synthetic()
    csv = "pokemon_image_dataset.csv"
    csv2 = "synthetic_pokemon.csv"
    #data_exploration(csv)
    pass