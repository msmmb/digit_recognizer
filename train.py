#!/usr/bin/env python3

from cnn import CNN

import gzip
import numpy as np
from tqdm import trange


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
    nn.layer1.filters.tofile("model/layer1.npy")
    nn.layer3.filters.tofile("model/layer3.npy")
    nn.layer5.filters.tofile("model/layer5.npy")
    nn.layer6.weights.tofile("model/layer6.npy")
    nn.layer7.weights.tofile("model/layer7.npy")
    

def train():
    nn = CNN()
    x_train, y_train, x_test, y_test = get_images()
    train_split_size = x_train.shape[0]
    test_split_size = x_test.shape[0]

    right_preds = 0
    for i in (t := trange(train_split_size)):
        y_pred = nn.forward(x_train[i])
        nn.backward(y_train[i], 5e-3)
        if np.argmax(y_pred) == np.argmax(y_train[i]):
            right_preds += 1
        t.set_description(f'train accuracy -> {round(right_preds/(i+1)*100, 2)} %')

    save_model(nn)

    right_preds = 0
    for i in (t := trange(test_split_size)):
        y_pred = np.argmax(nn.forward(x_test[i]))
        if y_pred == y_test[i]:
            right_preds += 1
        t.set_description(f'test accuracy -> {round(right_preds/(i+1)*100, 2)} %')

if __name__ == '__main__':
    train()
