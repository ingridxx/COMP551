import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Preprocess:

    dataset = []
    X = []
    y = []
    theta = 0
    data = []

    def __init__(self, dataset, class_variable):
        try:
            self.dataset = open(dataset, 'r')
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        clean = []
        temp = []

        for row in self.dataset:
            raw = row.split(',')
            for value in raw:
                try:
                    temp.append(float(value))
                except:
                    if value.split('\n')[0] == str(class_variable):
                        temp.append(1)
                    else:
                        temp.append(0)
                    clean.append(temp)
                    temp = []
                    continue

        self.dataset = clean
        temp = self.transpose(clean)
        self.X = np.array(self.transpose(temp[:-1]))
        self.y = np.array(temp[-1])

    def transpose(self, arr):
        return [[arr[j][i] for j in range(len(arr))] for i in range(len(arr[0]))]

    def add_bias(self, X):
        new_X = []
        for arr in X:
            arr.insert(0, 1)
            new_X.append(arr)
        return new_X

    def conv_to_col(self, y):
        new_y = []
        for item in y:
            item = [item]
            new_y.append(item)
        return new_y
