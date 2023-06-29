import numpy as np
from layers import Convolutional, MaxPool, Lineal


class CNN:

    def __init__(self):
        self.layer1 = Convolutional(5, 32, 28)
        self.layer2 = MaxPool(32, 2)
        self.layer3 = Convolutional(3, 32, 12)
        self.layer4 = MaxPool(32, 2)
        self.layer5 = Convolutional(2, 32, 5)
        self.layer6 = Lineal(32*4*4, 64)
        self.layer7 = Lineal(64, 10)

    def forward(self, x):
        self.input = x
        x, self.mask1 = self.reLU(self.layer1.forward(x, self.layer1.filters))
        x = self.layer2.forward(x)
        x, self.mask3 = self.reLU(self.layer3.forward(x, self.layer3.filters))
        x = self.layer4.forward(x)
        x, self.mask5 = self.reLU(self.layer5.forward(x, self.layer5.filters))
        x = np.reshape(x, (32*4*4, 1))
        x, self.mask6 = self.reLU(self.layer6.forward(x))
        y = self.softmax(self.layer7.forward(x))
        self.output = y
        return y

    def backward(self, y_true, learning_rate):
        dL_dX = self.softmax_backward(y_true)
        dL_dX = self.layer7.backward(dL_dX, learning_rate)
        dL_dX = self.reLU_backward(dL_dX, self.mask6)
        dL_dX = self.layer6.backward(dL_dX, learning_rate)
        dL_dX = dL_dX.reshape((32, 4, 4))
        dL_dX = self.reLU_backward(dL_dX, self.mask5)
        dL_dX = self.layer5.backward(dL_dX, learning_rate)
        dL_dX = self.layer4.backward(dL_dX)
        dL_dX = self.reLU_backward(dL_dX, self.mask3)
        dL_dX = self.layer3.backward(dL_dX, learning_rate)
        dL_dX = self.layer2.backward(dL_dX)
        dL_dX = self.reLU_backward(dL_dX, self.mask1)
        dL_dX = self.layer1.backward(dL_dX, learning_rate)
        return dL_dX
        
    def reLU(self, x):
        reLU_mask = (x > 0).astype(int)
        return x * reLU_mask, reLU_mask

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        self.softmax_output = exps/np.sum(exps, axis=0)
        return self.softmax_output

    def reLU_backward(self, dL_dY, mask):
        return dL_dY * mask

    def softmax_backward(self, y_true):
        dL_dX = self.output - y_true
        return dL_dX

