import argparse
import os
import shutil

import keras.backend as K
import numpy as np
import tensorflow as tf

from mnist_clf.dataset import load_mnist
from mnist_clf.model import create_model, save_as_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', default='checkpoints', help='local directory path to save the model')

parser.add_argument('--train-file', default='mnist',
                    help='either local directory path or cloud storage path to load MNIST dataset')
parser = parser.parse_args()


def main(parser):
    # Set Random Seed
    np.random.seed(0)
    tf.set_random_seed(0)

    # Reset Session
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)

    # Disable loading of learning nodes
    K.set_learning_phase(0)

    # Data
    train_x, train_y, test_x, test_y = load_mnist(parser.train_file)  # load_mnist('gs://anderson-mnist')
    model, arg_max = create_model()

    # Train
    model.fit(train_x, train_y, batch_size=32, epochs=1, verbose=1)

    # Save the model
    model_path = os.path.join(parser.job_dir, 'model')
    shutil.rmtree(model_path, ignore_errors=True)
    save_as_tensorflow(model, model_path, arg_max=arg_max)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    # Save the model to the Cloud
    # if parser.job_dir is not None:
    #     remote_path = os.path.join(parser.job_dir, 'model.h5')
    #     if not gfile.Exists(parser.job_dir):
    #         gfile.MakeDirs(parser.job_dir)
    #     with gfile.GFile('/tmp/model.h5', mode='rb') as f:
    #         with gfile.GFile(remote_path, mode='wb') as w:  # save the model to the cloud storage
    #             w.write(f.read())


if __name__ == '__main__':
    # train_x, train_y, test_x, test_y = load_mnist()
    # samples = create_sample(test_x, test_y)
    main(parser)
