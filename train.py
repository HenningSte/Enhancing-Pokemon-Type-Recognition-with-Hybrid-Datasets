import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from utils import load, plot_history
from models import custom_conv_net

train_dataset, val_dataset = load(csv="pokemon_image_dataset.csv", image_path="./images/pokemon_image_dataset")

train_dataset2, val_dataset2 = load(csv="synthetic_pokemon.csv", image_path="./synthetic_images")

model1 = custom_conv_net((120,120,4))

model2 = custom_conv_net((120,120,3))

histories = []

history1 = model1.fit(
    train_dataset,
    epochs = 20,
    validation_data = val_dataset,
    verbose = 1
)

histories.append(history1)

history2 = model2.fit(
    train_dataset2,
    epochs = 20,
    validation_data = val_dataset2,
    verbose = 1
)

histories.append(history2)

plot_history(histories)