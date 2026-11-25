import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import fashion_mnist

seed = 2026

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
 
def load_cv_data_fashion_mnist(seed=2026):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Select 10,000 images for cross-validation
    Xtr, _, ytr, _ = train_test_split(
        x_train, y_train,
        train_size=10000,
        stratify=y_train,
        random_state=seed
    )

    return (Xtr, ytr), (x_test, y_test)
 
def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out