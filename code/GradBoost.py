import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def make_gb_model(lr=0.1, max_depth=3, max_leaf_nodes=None,
                  min_samples_leaf=20, l2=0.0):

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=lr,
        max_iter=1,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2,
        warm_start=True,
        early_stopping=False, # Make our own stopping rule
        random_state=2026
    )


def run_cv_gb(Xtr, ytr, lr=0.1, max_depth=3,
              max_leaf_nodes=None, min_samples_leaf=20,
              l2=0.0, max_iter=500):

    """5-fold stratified CV with early stopping on mean validation loss.
    Returns the lowest mean validation loss and best boosting iteration.
    """

    cv = StratifiedKFold(5, shuffle=True, random_state=2026)
    folds = []

    for train_idx, val_idx in cv.split(Xtr, ytr):
        X_train, X_val = Xtr[train_idx], Xtr[val_idx]
        y_train, y_val = ytr[train_idx], ytr[val_idx]

        # Not necessary to standardize for decision trees but done here to keep method as identical as possible to FFNN and CNN methods
        mean, std = X_train.mean(), X_train.std()
        X_train = ((X_train - mean) / std).astype("float32").reshape(len(train_idx), -1)
        X_val = ((X_val - mean) / std).astype("float32").reshape(len(val_idx), -1)

        model = make_gb_model(lr, max_depth, max_leaf_nodes,
                              min_samples_leaf, l2)

        folds.append((model, X_train, y_train, X_val, y_val))

    best_loss, best_iter, patience = np.inf, 0, 0

    for iteration in range(1, max_iter + 1):
        losses = []

        for model, X_train, y_train, X_val, y_val in folds:
            # Add one boosting iteration without restarting
            model.set_params(max_iter=iteration)
            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_val) # gives probability for each of the ten classes
            losses.append(log_loss(y_val, probabilities) # cross-entropy loss
)

        mean_loss = np.mean(losses)
        if iteration % 10 == 0: #prints every tenth iteration
            print(f"Iteration {iteration}: mean val loss = {mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss, best_iter, patience = mean_loss, iteration, 0
        else:
            patience += 1

        if patience >= 10: # 10 instead of 3 because model can improve slowly
            break

    print(f"Stopped at iteration {iteration}, best iteration: {best_iter}")
    return float(best_loss), best_iter