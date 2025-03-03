from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train/255
    X_test = X_test/255
    return X_train, y_train, X_test, y_test

def plot_images(X_train, y_train):
    fig, ax = plt.subplots(1, np.unique(y_train).shape[0], figsize=(20, 20))
    for i in range(np.unique(y_train).shape[0]):
        ax[i].imshow(X_train[np.where(y_train == i)[0][0]], cmap='gray')
        ax[i].set_title(y_train[np.where(y_train == i)[0][0]])
        ax[i].axis('off')
    plt.show()
