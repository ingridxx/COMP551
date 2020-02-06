# Naive Baye's implementation
# Feel free to edit/change/delete anything, I honestly don't know if I did this right -Ali
import numpy as np
import pandas as pd
from preprocess import Preprocess
from math import sqrt
from math import pi
from math import exp


class NaiveBaye:

    # For binary classification (ionosphere and adult)
    # Code is from the lecture slides
    def bernoulli(self, prior, likelihood, X):
        logp = np.log(prior) + np.sum(np.log(likelihood) * X[:, None], 0) + np.sum(np.log(1 - likelihood) * (1 - X[:, None]), 0)
        log_p = np.max(log_p)
        posterior = np.exp(log_p)
        posterior /= np.sum(posterior)
        return posterior

    def separate_data(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean, standard deviation and count for each column in a dataset
    def summarize_dataset(self, dataset):
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(self, dataset):
        separated = self.separate_data(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x - mean)**2 / (2 * stdev**2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    def main(self):
        data = Preprocess("ionosphere.data")
        # print(data.dataset)
        # separated = self.separate_data(data.dataset)
        # summary = self.summarize_by_class(data.dataset)
        # for label in summary:
        #     print(label)
        #     for row in summary[label]:
        #         print(row)
        summaries = self.summarize_by_class(data.dataset)
        probabilities = self.calculate_class_probabilities(summaries, data.dataset[0])
        print(probabilities)


if __name__ == '__main__':
    nm = NaiveBaye()
    nm.main()
    # nm = NaiveBaye()
    # data = Preprocess("ionosphere.data")
    # nm.fit(data.data)
    # nm.predict(nm.prior_probability, nm.conditional_probability, data.X)
