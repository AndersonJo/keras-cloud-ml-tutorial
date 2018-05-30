import json

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Data
def load_mnist(path='mnist'):
    mnist = input_data.read_data_sets(path, one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    return train_x, train_y, test_x, test_y


def create_sample(data_x, data_y, export: str = 'sample.json'):
    samples = dict()
    count = 0
    for x, y in zip(data_x, data_y):
        label_idx = np.argmax(y)
        if label_idx == count:
            samples[label_idx] = (x, y)
            count += 1

        if count == 10:
            break

    # samples = [{'key': k.tolist(), 'output': v[1].tolist(), 'image': v[0].tolist()} for k, v in samples.items()]
    samples = [{'image': v[0].tolist()} for k, v in samples.items()]
    # samples = sorted(samples, key=lambda x: x['key'])

    with open(export, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample))
            f.write('\n')
    return samples
