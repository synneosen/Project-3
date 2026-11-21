import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

n = 100
sigma = 0.1
test_size = 0.2
seed = 2026

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
 
def load_fashion_mnist( val_size = 10000, seed=2026, flatten=True):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)

    # Canonical split: first 60 000 = training, last 10 000 = test
    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    if not flatten:  # for the CNN
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
 
def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out