# DA6401_AS1

This repository contains tools for training neural network models with extensive configuration options and hyperparameter optimization using Weights & Biases.

## Links
Wandb report link: https://api.wandb.ai/links/me21b138-indian-institute-of-technology-madras/h385vxtx

Github link: https://github.com/Shrey1617539/DA6401_AS1

## Overview

This repository contains tools for neural network training and hyperparameter optimization:

1. train.py: Main training script for building, training, and evaluating neural network models
2. data_loading_function.py: It contains all the functions to load the data and get visualization on it
3. model_training_funciton.py: It contains all the model training function and evaluation functions.
4. sweep.py: Script for automated hyperparameter optimization using wandb sweeps
5. sweep.yaml: All the hyper-parameter values you want to go through and sweep configurations. This has wandb supported structure

These scripts work together to provide a complete workflow for neural network experimentation, training, and optimization.

## Features

- Customizable neural network architecture with variable layers and variable neurons per layer.
- Various activation function like ReLU, tanh, sigmoid, identity.
- Different optimizers available like GD, SGD, Momentum, nag, RMSprop, adam, nadam
- Support for 2 type of loss functions Mean-Squared Loss and Cross-Entropy Loss
- Wandb for experiment tracking

## Requirements

- Python
- wandb
- argparse
- numpy
- keras.datasets
- matplotlib
- yaml
- sklearn.metrics
- seaborn

## Installation

    git clone https://github.com/Shrey1617539/DA6401_AS1

## Usage

### Training a Model

    # Basic usage with default parameters
    python train.py

    # Custom training configuration
    python train.py --wandb_entity your-entity --wandb_project your-project --dataset fashion_mnist --epochs 20 --batch_size 64 --optimizer adam

### Evaluating a Model

    # Train and then use the same to evaluate on validation data and test data
    python train.py --evaluate_validation True --evaluate_test True
    # Load a saved model and evaluate on test data
    python train.py --load_model model.npz --evaluate_test True

- Evaluation can be done in 2 ways by directly using a model or by training it and then evaluating that model itself.
- Evaluation can be done on 3 dataset train, validation and test.

### Running Hyperparameter Sweeps with wandb

Change sweep.yaml file according to your sweep requirements # Run a sweep with default settings
python sweep.py

    # Custom sweep configuration
    python sweep.py --wandb_entity your-entity --wandb_project your-project --count 50

## Command Line Arguments

### train.py Arguments

| Short Flag | Long Flag               | Type  | Default Value                                    | Description                                                               |
| ---------- | ----------------------- | ----- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| `-we`      | `--wandb_entity`        | str   | "me21b138-indian-institute-of-technology-madras" | Wandb Entity used to track experiments                                    |
| `-wp`      | `--wandb_project`       | str   | "my-awesome-project"                             | Project name for tracking in W&B dashboard                                |
| `-d`       | `--dataset`             | str   | "fashion_mnist"                                  | Dataset to use (choices: "mnist", "fashion_mnist")                        |
| `-e`       | `--epochs`              | int   | 15                                               | Number of epochs to train neural network                                  |
| `-b`       | `--batch_size`          | int   | 32                                               | Batch size for training                                                   |
| `-l`       | `--loss`                | str   | "cross_entropy"                                  | Loss function (choices: "mean_squared_error", "cross_entropy")            |
| `-o`       | `--optimizer`           | str   | "adam"                                           | Optimizer (choices: "sgd", "momentum", "nag", "rmsprop", "adam", "nadam") |
| `-lr`      | `--learning_rate`       | float | 0.00045...                                       | Learning rate for optimization                                            |
| `-m`       | `--momentum`            | float | 0.93026...                                       | Momentum used by momentum and nag optimizers                              |
| `-beta`    | `--beta`                | float | 0.01715...                                       | Beta used by rmsprop optimizer                                            |
| `-beta1`   | `--beta1`               | float | 0.99076...                                       | Beta1 used by adam and nadam optimizers                                   |
| `-beta2`   | `--beta2`               | float | 0.99693...                                       | Beta2 used by adam and nadam optimizers                                   |
| `-eps`     | `--epsilon`             | float | 0.00000...                                       | Epsilon used by optimizers                                                |
| `-w_d`     | `--weight_decay`        | float | 0.0                                              | Weight decay used by optimizers                                           |
| `-w_i`     | `--weight_init`         | str   | "Xavier"                                         | Weight initialization method (choices: "random", "Xavier")                |
| `-nhl`     | `--num_layers`          | int   | 4                                                | Number of hidden layers in feedforward neural network                     |
| `-sz`      | `--hidden_size`         | int   | 128                                              | Number of neurons in each hidden layer                                    |
| `-a`       | `--activation`          | str   | "tanh"                                           | Activation function (choices: "identity", "sigmoid", "tanh", "ReLU")      |
| `-lm`      | `--load_model`          | str   | ""                                               | Name of model to load (e.g., "model.npz")                                 |
| `-pi`      | `--plot_image`          | bool  | False                                            | Visualize classes of the training data                                    |
| `-mn`      | `--model_name`          | str   | ""                                               | Name to save the model (e.g., "model.npz")                                |
| `-etr`     | `--evaluate_training`   | bool  | False                                            | Evaluate on training data with confusion matrix                           |
| `-eva`     | `--evaluate_validation` | bool  | False                                            | Evaluate on validation data with confusion matrix                         |
| `-ete`     | `--evaluate_test`       | bool  | False                                            | Evaluate on test data with confusion matrix                               |

### sweep.py Arguments

| Short Flag | Long Flag         | Type | Default Value                                    | Description                                |
| ---------- | ----------------- | ---- | ------------------------------------------------ | ------------------------------------------ |
| `-we`      | `--wandb_entity`  | str  | "me21b138-indian-institute-of-technology-madras" | Wandb Entity for tracking experiments      |
| `-wp`      | `--wandb_project` | str  | "my-awesome-project"                             | Project name for tracking in W&B dashboard |
| `-c`       | `-count`          | int  | 100                                              | Maximum number of sweeps per agent         |

## How Code FLows Data

### Model Training Process (train.py)

1. Initializes wandb for experiment tracking
2. Loads and preprocesses the specified dataset
3. Splits data into training, validation, and test sets
4. Initializes model weights and biases based on specified architecture
5. Trains the model using the selected optimizer and hyperparameters
6. Saves the trained model parameters
7. Optionally evaluates the model on training, validation, and test sets

### Hyperparameter Optimization (sweep.py)

1. Loads sweep configuration from a YAML file
2. Initializes a wandb sweep with the provided configuration
3. Launches a wandb agent to run the sweep with different hyperparameter combinations

## Weights & Biases Integration

To use wandb features:

1. Create a wandb account if you don't have one
2. Log in to wandb: wandb login
3. Set your entity and project name in the command line arguments
