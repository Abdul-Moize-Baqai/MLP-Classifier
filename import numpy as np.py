import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# STEP 1 — PERCEPTRON CLASS
# ============================================================

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # ========================================================
    # STEP 2 — WEIGHTED SUM
    # ========================================================

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # ========================================================
    # STEP 3 — STEP FUNCTION
    # ========================================================

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    # ========================================================
    # STEP 4 — TRAINING
    # ========================================================

    def fit(self, X, y):

        # Initialize weights to zero
        # One extra weight is used for the bias
        self.w_ = np.zeros(1 + X.shape[1])

        # Store number of errors for each iteration
        self.errors_ = []

        print("Weights:", self.w_)

        # Repeat training n_iter times
        for _ in range(self.n_iter):

            # Error counter for this iteration
            error = 0

            # Go through every training example
            for xi, y in zip(X, y):

                # ------------------------------------------------
                # 1. Calculate prediction
                # ------------------------------------------------

                y_pred = self.predict(xi)

                # ------------------------------------------------
                # 2. Calculate weight update
                # ------------------------------------------------

                update = self.eta * (y - y_pred)

                # ------------------------------------------------
                # 3. Update feature weights
                # ------------------------------------------------

                self.w_[1:] = self.w_[1:] + update * xi

                # ------------------------------------------------
                # Update bias
                # ------------------------------------------------

                self.w_[0] = self.w_[0] + update

                # ------------------------------------------------
                # Count error if prediction was wrong
                # ------------------------------------------------

                error += int(update != 0.0)

            # Store errors for this iteration
            self.errors_.append(error)

        return self


# ============================================================
# STEP 5 — LOAD IRIS DATA
# ============================================================

# Iris dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load the dataset
df = pd.read_csv(url, header=None)

# Shuffle the dataset
df = shuffle(df)

# Display first 5 rows
print("Dataset:")
print(df.head())


# ============================================================
# SEPARATE X AND y
# ============================================================

# First four columns = features
X = df.iloc[:, 0:4].values

# Fifth column = target
y = df.iloc[:, 4].values

# Display first five feature rows
print("\nFirst 5 X values:")
print(X[0:5])

# Display first five labels
print("\nFirst 5 y values:")
print(y[0:5])


# ============================================================
# STEP 6 — SPLIT INTO TRAIN AND TEST
# ============================================================

train_data, test_data, train_labels, test_labels = train_test_split(
    X,
    y,
    test_size=0.25
)


# ============================================================
# CONVERT LABELS
# ============================================================

# Iris-setosa -> 1
# Everything else -> -1

train_labels = np.where(
    train_labels == "Iris-setosa",
    1,
    -1
)

test_labels = np.where(
    test_labels == "Iris-setosa",
    1,
    -1
)


# Display some training/testing data
print("\nTrain data:")
print(train_data[0:2])

print("\nTrain labels:")
print(train_labels[0:2])

print("\nTest data:")
print(test_data[0:2])

print("\nTest labels:")
print(test_labels[0:2])


# ============================================================
# STEP 6 — TRAIN THE PERCEPTRON
# ============================================================

from sklearn.linear_model import Perceptron

# Create scikit-learn Perceptron
perceptron = Perceptron(
    eta0=0.1,
    max_iter=10
)

# Train using training data and labels
perceptron.fit(
    train_data,
    train_labels
)


# ============================================================
# STEP 7 — MAKE PREDICTIONS
# ============================================================

test_preds = perceptron.predict(test_data)

print("\nTest predictions:")
print(test_preds)


# ============================================================
# STEP 8 — MEASURE PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    test_preds,
    test_labels
)

print(
    "Accuracy:",
    round(accuracy, 2) * 100,
    "%"
)