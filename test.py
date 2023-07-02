#!/usr/bin/env python3

from utils import load_model, test_model

if __name__ == '__main__':
    nn = load_model()
    test_model(nn)
