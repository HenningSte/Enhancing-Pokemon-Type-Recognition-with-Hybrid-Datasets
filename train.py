import tensorflow as tf
from utils import load, plot_history
from models import own_custom_conv_net, neuralmon_conv_net, smaller_VGGNet, custom_CNN_model
# parameter counts:     [1937682,            59026,             7738650,        282004]

# train on GPU if possible
tf.config.list_physical_devices('GPU')

# load and construct all datasets with load()
train_dataset, val_dataset = load(csv="pokemon_image_dataset.csv", image_path="./images/pokemon_image_dataset")

train_dataset2, val_dataset2 = load(csv="synthetic_pokemon.csv", image_path="./images/synthetic_pokemon")

train_dataset3, val_dataset3 = load(csv="synthetic_pokemon_v2.csv", image_path="./images/synthetic_pokemon_v2")

hybrid_train1 = train_dataset.concatenate(train_dataset2)

hybrid_val1 = val_dataset.concatenate(val_dataset2)

hybrid_train2 = train_dataset.concatenate(train_dataset3)

hybrid_val2 = val_dataset.concatenate(val_dataset3)


# define which model to use and for how many epochs to train
model = own_custom_conv_net

epochs = 150


model1 = model((120,120,3))

model2 = model((120,120,3))

model3 = model((120,120,3))

model4 = model((120,120,3))

model5 = model((120,120,3))


# optimize all models and save the histories
histories = []

history1 = model1.fit(
    train_dataset,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history1)
print("first model done")

history2 = model2.fit(
    train_dataset2,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history2)
print("second model done")

history3 = model3.fit(
    train_dataset3,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history3)
print("third model done")

history4 = model4.fit(
    hybrid_train1,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history4)
print("fourth model done")
plot_history([histories[3]])

history5 = model5.fit(
    hybrid_train2,
    epochs = epochs,
    validation_data = val_dataset,
    verbose = 1
)
histories.append(history5)
print("fifth model done")

plot_history(histories)