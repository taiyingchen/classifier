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

```python
from classifier import DecisionTree

X_train = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0],
            [3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [4.5, 3.0]]
y_train = [1, 1, 1, 3, 1, 3, 3, 3]
X_test = [[1.0, 2.2], [4.5, 1.0]]

classifier = DecisionTree(2)
classifier.fit(X_train, y_train)
y_test = classifier.predict(X_test)
print(y_test)

# [1, 1]
```
