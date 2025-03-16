from keras.datasets import fashion_mnist, mnist
import numpy as np
import matplotlib.pyplot as plt

# loading data with dataset name
def load_data(data_name = 'fashion_mnist'):
    if data_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    elif data_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalize the data before returning
    X_train = X_train/255
    X_test = X_test/255
    return X_train, y_train, X_test, y_test

# plotting 1 image per class 
def plot_images(X_train, y_train, num_cols=5):
    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    
    num_rows = int(np.ceil(num_classes / num_cols))
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    ax = np.array(ax).reshape(-1)
    
    for i, label in enumerate(unique_labels):
        indices = np.where(y_train == label)[0]
        if len(indices) > 0:
            img_idx = np.random.choice(indices)
            img = X_train[img_idx]
            cmap = 'gray' if len(img.shape) == 2 else None
            
            ax[i].imshow(img, cmap=cmap)
            ax[i].set_title(f"Class: {label}", fontsize=12)
            ax[i].axis('off')

    for j in range(i + 1, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()