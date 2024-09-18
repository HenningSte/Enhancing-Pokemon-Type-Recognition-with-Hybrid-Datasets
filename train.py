import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from utils import load, plot_history
from models import own_custom_conv_net, neuralmon_conv_net, smaller_VGGNet, custom_CNN_model

train_dataset, val_dataset = load(csv="pokemon_image_dataset.csv", image_path="./images/pokemon_image_dataset")

train_dataset2, val_dataset2 = load(csv="synthetic_pokemon.csv", image_path="./images/synthetic_pokemon")

train_dataset3, val_dataset3 = load(csv="synthetic_pokemon_v2.csv", image_path="./images/synthetic_pokemon_v2")

model1 = custom_CNN_model((120,120,4))

model2 = custom_CNN_model((120,120,3))

model3 = custom_CNN_model((120,120,3))

histories = []

history1 = model1.fit(
    train_dataset,
    epochs = 35,
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

history3 = model3.fit(
    train_dataset3,
    epochs = 20,
    validation_data = val_dataset3,
    verbose = 1
)
histories.append(history3)

plot_history(histories)