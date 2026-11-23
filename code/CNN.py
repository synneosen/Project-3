import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold

def make_model(lr=1e-3, dropout=0.5, kernel_size=3, conv_layers=2, dense_units=1024):   
    model_layers = [
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size, padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2))
    ]

    if conv_layers == 2:
        model_layers += [
            layers.Conv2D(64, kernel_size, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2))
        ]

    model_layers += [
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10)
    ]

    model = keras.Sequential(model_layers)
    
    model.compile(optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    
    return model



def run_cv(Xtr, ytr, lr=1e-3, dropout=0.5, kernel_size=3, conv_layers=2, dense_units=1024, epochs=30):
    print(f"\nLR={lr}, Dropout={dropout}, Kernel={kernel_size}, Conv layers={conv_layers}, Dense units={dense_units}")
    cv = StratifiedKFold(5, shuffle=True, random_state=2026)
    folds = []

    for train_idx, val_idx in cv.split(Xtr, ytr):
        X_train, X_val = Xtr[train_idx], Xtr[val_idx]
        y_train, y_val = ytr[train_idx], ytr[val_idx]

        mean, std = X_train.mean(), X_train.std()
        X_train = ((X_train - mean) / std).astype("float32")[..., None]
        X_val = ((X_val - mean) / std).astype("float32")[..., None]

        keras.utils.set_random_seed(2026)
        model = make_model(lr, dropout, kernel_size, conv_layers, dense_units)
        folds.append((model, X_train, y_train, X_val, y_val))

    best_loss, best_epoch, patience = np.inf, 0, 0

    for epoch in range(1, epochs + 1):
        losses = []

        for model, X_train, y_train, X_val, y_val in folds:
            history = model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=1, batch_size=64, verbose=0)
            losses.append(history.history["val_loss"][0])

        mean_loss = np.mean(losses)
        print(f"Epoch {epoch}: mean val loss = {mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss, best_epoch, patience = mean_loss, epoch, 0
        else:
            patience += 1

        if patience >= 3:
            break
    
    print(f"Stopped at epoch {epoch}, best epoch: {best_epoch}")
    assert epoch == epochs or epoch == best_epoch + 3
    return best_loss, best_epoch