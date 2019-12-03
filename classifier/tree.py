from sys import stdin
from utils import read_input
from math import log2
from collections import Counter


def _read_input():
    X_train = []
    y_train = []
    X_test = []
    
    for line in stdin:
        buffers = line.strip().split()
        label, attrs = int(buffers[0]), buffers[1:]

        X = []
        for attr in attrs:
            name, value = attr.split(':')
            name, value = int(name), float(value)
            X.append(value)

        if label == -1:  # Test instance
            X_test.append(X)
        else:  # Training instance
            X_train.append(X)
            y_train.append(label)

    return X_train, y_train, X_test


class DecisionTree():
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.depth = 0
        self.num_classes = 0

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError('Length of X and y cannot be 0')
        if len(X) != len(y):
            raise ValueError('Length of X does not match lenght of y')

        self.num_classes = len(set(y))
        self.num_data = len(X)
        self.num_attrs = len(X[0])

    def predict(self, X):
        pass

    def info(self, y):
        counter = Counter(y)
        total = len(y)
        probs = [count / total for count in counter.values()]
        ents = [-prob * log2(prob) for prob in probs]
        return sum(ents)

    def info_attr(self, y, y_left, y_right):
        return len(y_left) / len(y) * self.info(y_left) + len(y_right) / len(y) * self.info(y_right)

    def info_gain(self, y, y_left, y_right):
        return self.info(y) - self.info_attr(y, y_left, y_right)

    def find_best_split(self, X, y):
        max_info_gain = -float('inf')
        split_attr = -1
        best_split_point = -1

        # Iterate through attributes
        for i in range(self.num_attrs):
            attrs = [x[i] for x in X]
            # Sort continuous attribute values
            attrs_sorted = sorted(set(attrs))

            # Iterate through each split point
            for j in range(len(attrs_sorted)-1):
                split_point = (attrs_sorted[j] + attrs_sorted[j+1]) / 2
                left_indices = {index for index in range(len(X)) if X[index][i] <= split_point}
                # X_left = [X[index] for index in range(len(X)) if index in left_indices]
                # X_right = [X[index] for index in range(len(X)) if index not in left_indices]
                y_left = [y[index] for index in range(len(y)) if index in left_indices]
                y_right = [y[index] for index in range(len(y)) if index not in left_indices]
                gain = self.info_gain(y, y_left, y_right)
                if gain > max_info_gain:
                    max_info_gain = gain
                    best_split_point = split_point
                    split_attr = i
                print(gain)
        print(f'Max info gain: {max_info_gain}')
        print(f'Selected splitting attribute: {split_attr}')
        print(f'Best split point: {best_split_point}')

def main():
    X_train, y_train, X_test = read_input('input.txt')
    # print(X_train, y_train, X_test)

    classifier = DecisionTree(2)
    classifier.fit(X_train, y_train)
    # print(classifier.info(y_train))
    classifier.find_best_split(X_train, y_train)


if __name__ == '__main__':
    main()
