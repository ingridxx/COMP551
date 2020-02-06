import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Preprocess:

    dataset = []
    X = []
    y = []
    theta = 0
    data = []

    def __init__(self, dataset):
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
                    if value.split('\n')[0] == 'g':
                        temp.append(1)
                    else:
                        temp.append(0)
                    clean.append(temp)
                    temp = []
                    continue

        self.dataset = clean
        temp = self.transpose(clean)
        self.X = self.transpose(temp[:-1])
        self.data = self.X
        self.y = temp[-1]

        self.X = self.add_bias(self.X)
        self.y = self.conv_to_col(self.y)
        self.theta = np.zeros((len(self.X[0]), 1))

        for i in range(0, len(self.data)):
            self.data[i].append(self.y[i][0])

        columns = []
        for i in range(0, len(self.X[0]) - 1):
            columns.append(i)

        columns.append('y')
        self.data = pd.DataFrame(self.data, columns=columns)

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

    def main(self):
        for i in range(0, len(ionosphere.X)):
            # for j in range(0, len(ionosphere.X[i])):
            ionosphere.X[i].append(ionosphere.y[i][0])

        print(ionosphere.data)
        print(ionosphere.data)


if __name__ == '__main__':
    ionosphere = Preprocess("ionosphere.data")
    ionosphere.main()
