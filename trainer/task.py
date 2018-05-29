import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from mnist_clf.dataset import load_mnist, create_sample
from mnist_clf.model import create_model, save_model
import keras.backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mnist',
                    help='the path of training data (either local path or cloud storage path)')
parser.add_argument('--job-dir', default='checkpoints', help='local directory path to save the model')

parser = parser.parse_args()


def main(parser):
    # Set Random Seed
    np.random.seed(0)
    tf.set_random_seed(0)

    # Disable loading of learning nodes
    K.set_learning_phase(0)

    train_x, train_y, test_x, test_y = load_mnist('mnist')  # load_mnist('gs://anderson-mnist')
    model = create_model()

    # Train
    model.fit(train_x, train_y, epochs=1, verbose=1)

    # Save the model locally

    save_model(model, 'checkpoints/model.ckpt')

    # Save the model to the Cloud
    # if parser.job_dir is not None:
    #     remote_path = os.path.join(parser.job_dir, 'model.h5')
    #     if not gfile.Exists(parser.job_dir):
    #         gfile.MakeDirs(parser.job_dir)
    #     with gfile.GFile('/tmp/model.h5', mode='rb') as f:
    #         with gfile.GFile(remote_path, mode='wb') as w:  # save the model to the cloud storage
    #             w.write(f.read())

    # Evaluate
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)


if __name__ == '__main__':
    # train_x, train_y, test_x, test_y = load_mnist()
    # samples = create_sample(test_x, test_y)
    main(parser)
