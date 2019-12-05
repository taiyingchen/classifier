import csv


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


def read_csv(filename):
    X_train = []
    y_train = []
    X_test = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # Ignore header
                line_count += 1
            else:
                # Ignore animal name at row[0]
                row[1:] = [int(v) for v in row[1:]]
                if row[-1] == -1:
                    X_test.append(row[1:-1])
                else:
                    X_train.append(row[1:-1])
                    y_train.append(row[-1])
                line_count += 1
    return X_train, y_train, X_test
