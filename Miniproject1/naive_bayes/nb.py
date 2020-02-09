from preprocess import Preprocess
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class NaiveBayes:

    # input numpy nd array, first dimension is the number of samples, second dimension is the number of features (rows = samples, cols = features)
    # y is a 1-D row vector which is size of number of samples
    def fit(self, X, y):
        # Unpack
        n_samples, n_features = X.shape

        # Finds the number of unique elements in our output
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            # prior probability = frequency of the class / total number of samples
            self._prior[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    # Helper method for a single variable
    def _predict(self, x):
        predictions = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            predictions.append(posterior)

        # Choose the class for the highest probability
        return self._classes[np.argmax(posteriors)]

    # Probability density function (Gaussian)
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        try:
            denominator = np.sqrt(2 * np.pi * var)
            numarator = np.exp(- (x - mean)**2 / (2 * var))
            return numarator / denominator
        except RuntimeWarning:
            return 1

    def evaluate_acc(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y) * 100
        return accuracy


#np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("error")
nb = NaiveBayes()


# dataframe = Preprocess("ionosphere.data", "g")
# X = dataframe.X
# y = dataframe.y


dataframe = pd.read_csv("adult.data", header=None)
dataframe = dataframe.apply(LabelEncoder().fit_transform)
array = dataframe.values
X = array[:, :-1]
y = array[:, -1:]
y = y.ravel()

nb.fit(X, y)
predictions = nb.predict(X)

print("Accuracy in %", nb.evaluate_acc(y, predictions))
