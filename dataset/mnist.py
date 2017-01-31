import os
import pickle

import numpy as np
from PIL import Image


MNIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist.pkl')


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def to_one_hot(X, label_number=10):
    T = np.zeros((X.size, label_number))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """Load MNIST data set

    Parameters
    ----------
    normalize : Normalize image's pixel to 0~1.0
    flatten : If True, convert image matrices to vectors
    one_hot_label : If True, return one-hot labels. e.g. [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    Returns
    -------
    (Train image, Train label), (Test image, Test label)
    """
    if not os.path.exists(MNIST_FILE):
        raise OSError('Please generate dataset/mnist.pkl. See README for details.')

    with open(MNIST_FILE, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = to_one_hot(dataset['train_label'])
        dataset['test_label'] = to_one_hot(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    (x_train, y_train), (_, _) = load_mnist(normalize=False)
    img = x_train[0].reshape((28, 28))
    label = y_train[0]
    print(label)
    img_show(img)
