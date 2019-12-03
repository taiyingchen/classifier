def read_input(filename):
    X_train = []
    y_train = []
    X_test = []

    with open(filename, 'r') as file:
        for line in file:
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
