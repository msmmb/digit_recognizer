import sys
import gzip
import signal
import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN


def get_images():
    with gzip.open('mnist/train-images-idx3-ubyte.gz', 'rb') as data:
        data.seek(4)
        train_split_size = int.from_bytes(data.read(4), 'big')
        rows = int.from_bytes(data.read(4), 'big')
        cols = int.from_bytes(data.read(4), 'big')
        train_images = data.read()
        x_train = np.frombuffer(train_images, dtype=np.uint8)
        x_train = x_train.reshape((train_split_size, rows, cols))

    with gzip.open('mnist/train-labels-idx1-ubyte.gz', 'rb') as data:
        train_labels = data.read()[8:]
        train_labels = np.frombuffer(train_labels, dtype=np.uint8)
        y_train = np.zeros((train_split_size, 10, 1))
        for i in range(train_split_size):
            y_train[i, train_labels[i]] = 1

    with gzip.open('mnist/t10k-images-idx3-ubyte.gz', 'rb') as data:
        data.seek(4)
        test_split_size = int.from_bytes(data.read(4), 'big')
        rows = int.from_bytes(data.read(4), 'big')
        cols = int.from_bytes(data.read(4), 'big')
        test_images = data.read()
        x_test = np.frombuffer(test_images, dtype=np.uint8)
        x_test = x_test.reshape((test_split_size, rows, cols))

    with gzip.open('mnist/t10k-labels-idx1-ubyte.gz', 'rb') as data:
        test_labels = data.read()[8:]
        y_test = np.frombuffer(test_labels, dtype=np.uint8)

    return x_train/255, y_train, x_test/255, y_test


def save_model(nn):
    nn.layer1.filters.tofile('model/layer1_filters.npy')
    nn.layer1.biases.tofile('model/layer1_biases.npy')
    nn.layer3.filters.tofile('model/layer3_filters.npy')
    nn.layer3.biases.tofile('model/layer3_biases.npy')
    nn.layer5.filters.tofile('model/layer5_filters.npy')
    nn.layer5.biases.tofile('model/layer5_biases.npy')
    nn.layer6.weights.tofile('model/layer6_weights.npy')
    nn.layer6.biases.tofile('model/layer6_biases.npy')
    nn.layer7.weights.tofile('model/layer7_weights.npy')
    nn.layer7.biases.tofile('model/layer7_biases.npy')


def load_model():
    nn = CNN()
    nn.layer1.filters = np.reshape(np.fromfile('model/layer1_filters.npy'), nn.layer1.filters.shape)
    nn.layer1.biases = np.reshape(np.fromfile('model/layer1_biases.npy'), nn.layer1.biases.shape)
    nn.layer3.filters = np.reshape(np.fromfile('model/layer3_filters.npy'), nn.layer3.filters.shape)
    nn.layer3.biases = np.reshape(np.fromfile('model/layer3_biases.npy'), nn.layer3.biases.shape)
    nn.layer5.filters = np.reshape(np.fromfile('model/layer5_filters.npy'), nn.layer5.filters.shape)
    nn.layer5.biases = np.reshape(np.fromfile('model/layer5_biases.npy'), nn.layer5.biases.shape)
    nn.layer6.weights = np.reshape(np.fromfile('model/layer6_weights.npy'), nn.layer6.weights.shape)
    nn.layer6.biases = np.reshape(np.fromfile('model/layer6_biases.npy'), nn.layer6.biases.shape)
    nn.layer7.weights = np.reshape(np.fromfile('model/layer7_weights.npy'), nn.layer7.weights.shape)
    nn.layer7.biases = np.reshape(np.fromfile('model/layer7_biases.npy'), nn.layer7.biases.shape)
    return nn


def signal_handler(sig, frame):
    print('\nProgram exited succesfully.')
    sys.exit(0)


def test_model(nn):
    _, _, x_test, y_test = get_images()
    print('Press Ctrl-C to finish\n')
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        n = np.random.randint(0, x_test.shape[0])
        y_pred = nn.forward(x_test[n])
        print(f'Predicted digit: {np.argmax(y_pred)}')
        print(f'Actual digit: {y_test[n]}')

        for i in range(10):
            prob = f'{int(y_pred[i][0]*100)}'
            if prob != '0':
                print(f'{i} -> {prob}%')
                
        print()
        plt.imshow(x_test[n])
        plt.axis('off')
        plt.show()
