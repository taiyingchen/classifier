from math import log2
from collections import Counter


class Node():
    def __init__(self, depth):
        self.depth = depth
        self.attr = None
        self.split_point = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.leaf_class = None


class DecisionTree():
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError('Length of X and y cannot be 0')
        if len(X) != len(y):
            raise ValueError('Length of X does not match lenght of y')

        self.num_classes = len(set(y))
        self.num_attrs = len(X[0])
        self.root = Node(0)

        self.fit_helper(X, y, self.root)

    def fit_helper(self, X, y, node):
        # Stopping criteria
        if node.depth >= self.max_depth or len(set(y)) <= 1:
            node.is_leaf = True
            node.leaf_class = max(Counter(y).most_common(), key=lambda v: v[1])[0]
        else:
            split_attr, split_point = self.find_best_split(X, y)
            X_left, X_right, y_left, y_right = self.split(X, y, split_attr, split_point)
            node.attr = split_attr
            node.split_point = split_point
            depth = node.depth
            if len(X_left) > 0:
                node.left = Node(depth+1)
                self.fit_helper(X_left, y_left, node.left)
            if len(X_right) > 0:
                node.right = Node(depth+1)
                self.fit_helper(X_right, y_right, node.right)

    def predict(self, X):
        y = []
        for Xi in X:
            y.append(self.predict_helper(Xi))
        return y

    def predict_helper(self, X):
        node = self.root
        while not node.is_leaf:
            if X[node.attr] <= node.split_point:
                node = node.left
            else:
                node = node.right
        return node.leaf_class

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
        for attr in range(self.num_attrs):
            attrs = [x[attr] for x in X]
            # Sort continuous attribute values
            attrs_sorted = sorted(set(attrs))

            # Iterate through each split point
            for j in range(len(attrs_sorted)-1):
                split_point = (attrs_sorted[j] + attrs_sorted[j+1]) / 2
                _, _, y_left, y_right = self.split(X, y, attr, split_point)
                gain = self.info_gain(y, y_left, y_right)
                if gain > max_info_gain:
                    max_info_gain = gain
                    best_split_point = split_point
                    split_attr = attr
        
        return split_attr, best_split_point

    def split(self, X, y, split_attr, split_point):
        left_indices = {index for index in range(len(X)) if X[index][split_attr] <= split_point}
        X_left = [X[index] for index in range(len(X)) if index in left_indices]
        X_right = [X[index] for index in range(len(X)) if index not in left_indices]
        y_left = [y[index] for index in range(len(y)) if index in left_indices]
        y_right = [y[index] for index in range(len(y)) if index not in left_indices]
        return X_left, X_right, y_left, y_right


def main():
    X_train = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0],
               [3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [4.5, 3.0]]
    y_train = [1, 1, 1, 3, 1, 3, 3, 3]
    X_test = [[1.0, 2.2], [4.5, 1.0]]

    clf = DecisionTree(2)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    print(y_test)


if __name__ == '__main__':
    main()
