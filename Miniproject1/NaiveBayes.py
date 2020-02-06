# Naive Baye's implementation
# Feel free to edit/change/delete anything, I honestly don't know if I did this right -Ali
import numpy as np
import pandas as pd
from preprocess import Preprocess
from random import seed
from random import randrange
from math import sqrt
from math import pi
from math import exp


class NaiveBaye:

    # # For binary classification (ionosphere and adult)
    # # Code is from the lecture slides
    # def bernoulli(self, prior, likelihood, X):
    #     logp = np.log(prior) + np.sum(np.log(likelihood) * X[:, None], 0) + np.sum(np.log(1 - likelihood) * (1 - X[:, None]), 0)
    #     log_p = np.max(log_p)
    #     posterior = np.exp(log_p)
    #     posterior /= np.sum(posterior)
    #     return posterior

    def separate_data(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the following for each data point:
    #   - mean
    #   - standard deviation
    #   - count
    def dataset_statistics(self, dataset):
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

    # Statistics for each row
    def class_statistics(self, dataset):
        separated = self.separate_data(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.dataset_statistics(rows)
        return summaries

    def probability(self, x, mean, stdev):
        # Stdev calculation is not accurate for very small numbers
        if stdev == 0:
            #stdev = 0.0001
            return 1
        exponent = exp(-((x - mean)**2 / (2 * stdev**2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def likelihood(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= self.probability(row[i], mean, stdev)
        return probabilities

    def predict(self, summaries, row):
        probabilities = self.likelihood(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    # Naive Bayes Algorithm
    def fit(self, train, test):
        summarize = self.class_statistics(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)

    # Calculate accuracy percentage
    def evaluate_accuracy(self, y, y_hat):
        correct_predictions = 0
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                correct_predictions += 1
        return correct_predictions / float(len(y)) * 100.0

    # The following evaluation techniques are from this resource: < will cite it lol >
    # --------------------------------------------------------------------------------#
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.evaluate_accuracy(actual, predicted)
            scores.append(accuracy)
        return scores
    # --------------------------------------------------------------------------------#

    def main(self):
        data = Preprocess("ionosphere.data")
        n_folds = 5
        scores = self.evaluate_algorithm(data.dataset, self.fit, n_folds)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
    nm = NaiveBaye()
    nm.main()
    # nm = NaiveBaye()
    # data = Preprocess("ionosphere.data")
    # nm.fit(data.data)
    # nm.predict(nm.prior_probability, nm.conditional_probability, data.X)
