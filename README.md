# Enhancing Pokémon Type Recognition with Hybrid Datasets
This project explores the use of hybrid datasets to improve Pokémon type recognition through deep learning models. It is part of the "Enhancing AI with Hybrid Datasets" course at Osnabrück University (SuSe 2024). The goal is to evaluate different convolutional neural network architectures using both real and synthetic data.
<small>by Felix Japtok, Eva Kuth, Imogen Hüsing, Henning Stegemann</small>

## Prerequisites
Make sure you have the following installed:
- Python 3.x

All other dependencies can be installed as described in the next section


## How to run the Experiment
1. Clone this repository
```bash
git clone https://github.com/HenningSte/Enhancing-Pokemon-Type-Recognition-with-Hybrid-Datasets
```
2. Navigate to the installed repository
```bash 
cd  Enhancing-Pokemon-Type-Recognition-with-Hybrid-Datasets
```
3. Install the necessary Python packages
```bash
pip install -r requirements.txt
```
4. Edit **train.py** to choose which of the implemented models to use (listed in line 3). You can modify the model in line 26
```python
from models import own_custom_conv_net, neuralmon_conv_net, smaller_VGGNet, custom_CNN_model

model = own_custom_conv_net  # Example choice
```
5. Adjust the number of epochs (line 28)
```python
epochs = 50
```
6. Run the training script from your console
```bash
python train.py
```

## Repository Structure
- **/images**: Contains all images used in the dataset and screenshots from the stable diffusion structure.
- **pokemon_image_dataset.csv**, **synthetic_pokemon.csv**, **synthetic_pokemon_v2.csv**: CSV files containing labels for the datasets.
- **/plots**: Includes all plots used in the report and additional plots related to model trainability.
- **model_analysis.ipynb**: Jupyter notebook for analyzing the models used.
- **model_and_dataset_analysis.ipynb**: Detailed analysis of both the model performance and datasets used.
- **train.py**: The main script to train the selected model on the dataset.