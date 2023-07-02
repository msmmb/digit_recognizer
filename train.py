#!/usr/bin/env python3

import numpy as np
from tqdm import trange

from cnn import CNN
from utils import get_images, save_model


if __name__ == '__main__':
    nn = CNN()
    num_epochs = 3
    x_train, y_train, x_test, y_test = get_images()
    train_split_size = x_train.shape[0]
    test_split_size = x_test.shape[0]

    for epoch in range(num_epochs):
        print(f'\nEPOCH {epoch+1}/{num_epochs}')
        right_preds = 0
        for i in (t := trange(train_split_size)):
            y_pred = nn.forward(x_train[i])
            nn.backward(y_train[i], 1e-2)
            if np.argmax(y_pred) == np.argmax(y_train[i]):
                right_preds += 1
            t.set_description(f'train accuracy -> {round(right_preds/(i+1)*100, 2)}%')

        save_model(nn)

    print()

    right_preds = 0
    for i in (t := trange(test_split_size)):
        y_pred = np.argmax(nn.forward(x_test[i]))
        if y_pred == y_test[i]:
            right_preds += 1
        t.set_description(f'test accuracy -> {round(right_preds/(i+1)*100, 2)}%')

