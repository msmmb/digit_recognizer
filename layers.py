import numpy as np


class Convolutional:

    def __init__(self, kernel_size, depth, input_size):
        output_size = input_size-kernel_size+1
        self.filters = np.random.rand(depth, kernel_size, kernel_size) 
        self.biases = np.zeros((depth, output_size, output_size))
        
    def get_region(self, x, kernel_size, output_size):
        for i in range(x.shape[0]):
            for j in range(output_size):
                for k in range(output_size):
                    region = x[i, j:j+kernel_size, k:k+kernel_size]
                    yield region, i, j, k
    
    def conv(self, x, filters):
        output_size = x.shape[1]-filters.shape[1]+1
        y = np.zeros((x.shape[0], output_size, output_size), dtype=np.float128)
        for region, i, j, k in self.get_region(x, filters.shape[1], output_size):
            y[i, j, k] = np.sum(region*filters[i])
        return y

    def forward(self, x, filters):
        self.input = x
        y = self.conv(x, filters)+self.biases
        self.output = y
        return y

    def backward(self, dL_dY, learning_rate):
        rotated_filters = self.filters.transpose(0,2,1)
        pad_size = self.filters.shape[1]-1
        padded_dL_dY = np.array(list(map(lambda f: np.pad(dL_dY[f], pad_size), range(dL_dY.shape[0]))))

        dL_dW = self.conv(self.input, dL_dY)
        dL_dX = self.conv(padded_dL_dY, rotated_filters)
        self.filters -= dL_dW*learning_rate
        self.biases -= dL_dY*learning_rate
        return dL_dX


class Lineal:

    def __init__(self, x):
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
    
    def forward(self, x):
        self.input = x
        y = np.dot(self.weights, self.input) + self.biases
        self.output = y
        return y

    def backward(self, dL_dY, learning_rate):
        dL_dW = np.dot(dL_dY, self.input.T)
        dL_dX = np.dot(self.weights.T, dL_dY)
        self.weights -= dL_dW*learning_rate
        self.biases -= dL_dY*learning_rate
        return dL_dX
