from mnist_clf.dataset import load_mnist, create_sample

train_x, train_y, test_x, test_y = load_mnist()
samples = create_sample(test_x, test_y)
