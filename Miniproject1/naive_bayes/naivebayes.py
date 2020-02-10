from preprocess import Preprocess
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class NaiveBayes:

    _mean = []
    _stdvar = []
    _prior_probability = []

    def predict(self, X):
        y_hat = [self.predict_single(x) for x in X]
        return y_hat

    # Helper method for a single variable
    def predict_single(self, x):
        predictions = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior_probability[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            predictions.append(posterior)

        # Choose the class for the highest probability
        return self._classes[np.argmax(predictions)]

    # Probability density function (Gaussian)
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._stdvar[class_idx]
        try:
            denominator = np.sqrt(2 * np.pi * var)
            numarator = np.exp(- (x - mean)**2 / (2 * var))
            if np.all(numarator / denominator != 0):
                return numarator / denominator
            else:
                return 1
        except RuntimeWarning:
            return 1

    # X: numpy nd array, rows = samples, columns = features
    # y: 1-D row vector, size of number of samples
    def fit(self, X, y):
        # Unpack
        n_samples, n_features = X.shape

        # Finds the number of unique elements in our output
        self._classes = np.unique(y)

        n_classes = len(self._classes)
        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._stdvar = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior_probability = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._stdvar[c, :] = X_c.var(axis=0)

            # prior probability = frequency / total number of samples
            self._prior_probability[c] = X_c.shape[0] / float(n_samples)

    def evaluate_acc(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y) * 100
        return accuracy


# ----------------------------------------------------------------------------------- #
warnings.filterwarnings("error")
nb = NaiveBayes()

# Ionosphere
dataframe = Preprocess("data/ionosphere.data", "g")
X = dataframe.X
y = dataframe.y
history = nb.fit(X, y)
predictions = nb.predict(X)
print("Accuracy in % for ionosphere dataset: ", nb.evaluate_acc(y, predictions))

# # Adult
# dataframe = pd.read_csv("data/adult.data", header=None)
# dataframe = dataframe.apply(LabelEncoder().fit_transform)
# array = dataframe.values
# X = array[:, :-1]
# y = array[:, -1:]
# y = y.ravel()
# nb.fit(X, y)
# predictions = nb.predict(X)
# print("Accuracy in % for adult dataset: ", nb.evaluate_acc(y, predictions))

# # Breast-Cancer
# dataframe = pd.read_csv("data/breast-cancer.data", header=None)
# dataframe = dataframe.replace(to_replace="?", value=np.nan)
# dataframe = dataframe.dropna()
# dataframe = dataframe.apply(LabelEncoder().fit_transform)
# array = dataframe.values
# X = array[:, :-1]
# y = array[:, -1:]
# y = y.ravel()
# nb.fit(X, y)
# predictions = nb.predict(X)
# print("Accuracy in % for breast-cancer dataset: ", nb.evaluate_acc(y, predictions))

# # Wine
# dataframe = pd.read_csv("data/winequality-red.csv", sep=';', header=None, skiprows=1)
# array = dataframe.values
# X = array[:, :-1]
# y = array[:, -1:]
# y = y.ravel()
# y = LabelEncoder().fit_transform(y)
# nb.fit(X, y)
# predictions = nb.predict(X)
# print("Accuracy in % for wine-quality dataset: ", nb.evaluate_acc(y, predictions))
# ----------------------------------------------------------------------------------- #
