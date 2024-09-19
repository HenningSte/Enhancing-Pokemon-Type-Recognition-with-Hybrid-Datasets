import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from utils import load, plot_history
from models import own_custom_conv_net, neuralmon_conv_net, smaller_VGGNet, custom_CNN_model

train_dataset, val_dataset = load(csv="pokemon_image_dataset.csv", image_path="./images/pokemon_image_dataset")

train_dataset2, val_dataset2 = load(csv="synthetic_pokemon.csv", image_path="./images/synthetic_pokemon")

train_dataset3, val_dataset3 = load(csv="synthetic_pokemon_v2.csv", image_path="./images/synthetic_pokemon_v2")

hybrid_train1 = train_dataset.concatenate(train_dataset2)

hybrid_val1 = val_dataset.concatenate(val_dataset2)

hybrid_train2 = train_dataset.concatenate(train_dataset3)

hybrid_val2 = val_dataset.concatenate(val_dataset3)



model = custom_CNN_model

epochs = 50



model1 = model((120,120,3))

model2 = model((120,120,3))

model3 = model((120,120,3))

model4 = model((120,120,3))

model5 = model((120,120,3))

histories = []

history1 = model1.fit(
    train_dataset,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history1)
print("first model done")
plot_history([histories[0]])

history2 = model2.fit(
    train_dataset2,
    epochs = epochs,
    validation_data = val_dataset2,
    verbose = 1
)
histories.append(history2)
print("second model done")
plot_history([histories[1]])

history3 = model3.fit(
    train_dataset3,
    epochs = epochs,
    validation_data = val_dataset3,
    verbose = 1
)
histories.append(history3)
print("third model done")
plot_history([histories[2]])

history4 = model4.fit(
    hybrid_train1,
    epochs = epochs,
    validation_data = hybrid_val1,
    verbose = 1
)
histories.append(history4)
print("fourth model done")
plot_history([histories[3]])

history5 = model5.fit(
    hybrid_train2,
    epochs = epochs,
    validation_data = hybrid_val2,
    verbose = 1
)
histories.append(history5)
print("fifth model done")
plot_history([histories[4]])

plot_history(histories)