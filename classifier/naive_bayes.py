from collections import defaultdict, Counter


class NaiveBayes():
    def __init__(self, smoothing_factor, num_classes):
        self.smoothing_factor = smoothing_factor
        self.num_classes = num_classes
        self.prob_y = {}
        self.prob_x_given_y = defaultdict(list)

    def fit(self, X, y, num_attr_values=None):
        classes = Counter(y)
        y2X = defaultdict(list)
        for i in range(len(y)):
            y2X[y[i]].append(X[i])

        self.num_attrs = len(X[0])

        for c, count in classes.items():
            self.prob_y[c] = (count + self.smoothing_factor) / \
                (len(y) + self.smoothing_factor * self.num_classes)

        for c, X_list in y2X.items():
            for attr in range(self.num_attrs):
                attr_count = []
                for i in range(len(X_list)):
                    attr_count.append(X_list[i][attr])
                attr_count = Counter(attr_count)
                prob_values = {}
                for attr_value in num_attr_values[attr]:
                    prob_values[attr_value] = (
                        attr_count[attr_value] + self.smoothing_factor) / (len(X_list) + self.smoothing_factor * len(num_attr_values[attr]))
                self.prob_x_given_y[c].append(prob_values)

    def predict(self, X):
        y = []
        for Xi in X:
            max_prob = -float('inf')
            for yi in self.prob_y:
                prob = self.joint_prob(Xi, yi)
                if prob > max_prob:
                    max_prob = prob
                    y_predict = yi
            y.append(y_predict)
        return y

    def joint_prob(self, X, y):
        prob = self.prob_y[y]
        for attr, xi in enumerate(X):
            prob *= self.prob_x_given_y[y][attr][xi]
        return prob


def main():
    X_train = [
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 6, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 6, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 2, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    ]
    y_train = [1, 7, 4, 6, 1, 6, 2, 7, 1, 3]
    X_test = [
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    ]

    num_attr_values = [
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        [0, 1], [0, 1], [0, 1], [0, 2, 4, 5, 6, 8], [0, 1], [0, 1], [0, 1]]
    num_classes = 7
    smoothing_factor = 0.1

    clf = NaiveBayes(smoothing_factor, num_classes)
    clf.fit(X_train, y_train, num_attr_values)
    y_test = clf.predict(X_test)
    print(y_test)


if __name__ == "__main__":
    main()
