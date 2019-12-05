# Classifier

Python implementation of Decision Tree and Naive Bayes classifiers

## Installation

```bash
git clone https://github.com/taiyingchen/classifier.git
cd classifier/
pip3 install .
```

### Dependencies

Required only Python (>= 3.6)

## Usage

### Decision Tree Classifier

```python
from classifier import DecisionTree

X_train = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0],
            [3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [4.5, 3.0]]
y_train = [1, 1, 1, 3, 1, 3, 3, 3]
X_test = [[1.0, 2.2], [4.5, 1.0]]

clf = DecisionTree(2)
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
print(y_test)
# [1, 1]
```

### Naive Bayes Classifier

```python
from classifier import NaiveBayes

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
# Specify unique values for each attribute
# e.g. All of the attribute are binary 
# Expect one have values: [0, 2, 4, 5, 6, 8]
num_attr_values = [
    [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    [0, 1], [0, 1], [0, 1], [0, 2, 4, 5, 6, 8], [0, 1], [0, 1], [0, 1]]
num_classes = 7
smoothing_factor = 0.1

clf = NaiveBayes(smoothing_factor, num_classes)
clf.fit(X_train, y_train, num_attr_values)
y_test = clf.predict(X_test)
print(y_test)
# [4]
```