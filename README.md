[TOC]

# Overview

Keras를 Google Cloud ML Engine에서 학습시키고 서빙까지 하는 방법을 튜토리얼로 제공합니다. 



# Tutorial

## Structure

Google Cloud ML Engine에서 사용하기 위해서는 다음과 같은 구조를 갖고 있어야 합니다. 

```
.
├── config.yaml
├── README.md
├── setup.py
└── trainer
    ├── __init__.py
    └── train.py
```



## Virtual Environtment

먼저 virtual environment 를 생성합니다. 

> 만약 Python2.7을 기본으로 사용하면 굳이 해줄필요는 없습니다. 
> gcloud ml-engine 명령어가 python 2.7 을 기본으로 사용하고 있기 때문입니다. 

```bash
virtualenv cmle --python=/usr/local/bin/python3.6 --system-site-packages
source cmle/bin/activate
```



## Google Cloud SDK

[Google Cloud SDK 설치방법](https://github.com/AndersonJo/google-cloud-platform/blob/master/01-quickstart.md) 을 참고하여 설치를 합니다. 

설치후 다음과 같은 명령어를 사용할 수 있습니다. 

**모델 리스트**

```bash
gcloud ml-engine models list
```



## MNIST Dataset

먼저 MNIST 데이터를 다운 받습니다.

```bash
mkdir mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

데이터를 Google Cloud Storage에 bucket을 생성하고 업로드를 합니다. 

```bash
gsutil mb gs://anderson-mnist
gsutil cp * gs://anderson-mnist
```

잘 올라갔는지 확인합니다.

```bash
gsutil ls gs://anderson-mnist
```



## Model 

먼저 모델을 넣을 디렉토리를 만듭니다.

```
mkdir trainer
```

model.py 안에 다음과 같이 Keras MNIST classification model을 만듭니다.

```python
import argparse
import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
from tensorflow.python.training.momentum import MomentumOptimizer

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mnist',
                    help='the path of training data (either local path or cloud storage path)')
parser.add_argument('--checkpoint', default='checkpoints', help='local directory path to save the model')
parser.add_argument('--cloud-path', help='directory path or cloud storage bucket to save the model or checkpoints')
parser = parser.parse_args()


# Data
def load_mnist(path='mnist'):
    mnist = input_data.read_data_sets(path, one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels

    return train_x, train_y, test_x, test_y


# Model
def create_model():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2, seed=0))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2, seed=0))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=MomentumOptimizer(0.01, momentum=0.9),
                  metrics=['accuracy'])
    return model


def save_to_cloud(local_path, cloud_path):
    # Save the model in the cloud
    if cloud_path is not None:
        remote_path = os.path.join(cloud_path, 'model.h5')
        if not gfile.Exists(cloud_path):
            gfile.MakeDirs(cloud_path)
        with gfile.GFile(local_path, mode='rb') as f:
            with gfile.GFile(remote_path, mode='wb') as w:  # save the model to the cloud storage
                w.write(f.read())


def main(parser):
    # Set Random Seed
    np.random.seed(0)
    tf.set_random_seed(0)

    train_x, train_y, test_x, test_y = load_mnist('gs://anderson-mnist')
    model = create_model()

    # Train
    model.fit(train_x, train_y, epochs=1, verbose=1)

    # Save the model locally
    local_path = os.path.join(parser.checkpoint, 'model.h5')
    if not os.path.exists(parser.checkpoint):
        os.makedirs(parser.checkpoint)
    model.save(local_path)

    # Save the model to the Cloud
    save_to_cloud(local_path, parser.cloud_path)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)


if __name__ == '__main__':
    main(parser)
```



로컬 환경에서 테스트를 합니다.

```bash
python3.6 trainer/train.py --checkpoint=checkpoints
```

Cloud Storage에 저장이 잘 되는지 테스트 

```bash
python3.6 trainer/train.py --job-dir=gs://anderson-mnist/checkpoints
```



## config.yaml 설정

Cloud ML Engine내에서 학습시킬때 Python 3.5를 사용하게 하거나 (default는 python2.7), GPU를 사용하기 위해서는 `config.yaml` 같은 파일을 만들고 설정파일을 넣습니다. 



예제로 다음과 같이 설정 할 수 있습니다.

```
trainingInput:
  pythonVersion: "3.5"
  scaleTier: CUSTOM
  masterType: standard_p100
  workerType: standard_p100
  parameterServerType: large_model
  workerCount: 0
  parameterServerCount: 0
```



#### Python 3.5 사용

아래의 설정이 들어가야 합니다.

```
trainingInput:
  pythonVersion: "3.5"
```

`gcloud ml-engine ` 을 사용시 ```--runtime-version``` 은 1.4 이상이 되야 합니다.
자세한 정보는 [Runtime Version List](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list) 를 참고 합니다. 

```
gcloud ml-engine jobs submit training $JOB_NAME \
  ...
  --runtime-version 1.8 \
```



#### GPU 설정

GPU 모델은 다음의 옵션으로 설정 가능합니다.

- `standard_gpu`: A single NVIDIA Tesla K80 GPU
- `complex_model_m_gpu`: Four NVIDIA Tesla K80 GPUs
- `complex_model_l_gpu`: Eight NVIDIA Tesla K80 GPUs
- `standard_p100`: A single NVIDIA Tesla P100 GPU (*Beta*)
- `complex_model_m_p100`: Four NVIDIA Tesla P100 GPUs (*Beta*)

현재 GPU는 해당 리젼에서만 제공이 됩니다.

- `us-east1`
- `us-central1`
- `asia-east1`
- `europe-west1`



## setup.py 설정하기

아래와 같이 설정을 합니다.

```python
from setuptools import setup, find_packages

setup(
    name='keras-google-cloud-machine-learning',
    version='0.1',
    packages=find_packages(),  # ['trainer'],
    url='',
    license='MIT',
    author='anderson',
    author_email='a141890@gmail.com',
    description='',
    install_requires=[
        'keras',
        'h5py'
    ]
)
```

실제 cloud ml engine상에서 돌려보면 필요한 라이브러리를 설치한 후에 실행이 됩니다. 
이때 참고 되는 파일이 setup.py입니다.



## gcloud 사용해서 학습시키기

먼저 변수들을 선언해줍니다.

```bash
export BUCKET_NAME=anderson-mnist
export JOB_NAME="mnist_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$CHECKPOINT
export REGION=us-east1
```

Local에서 학습을 테스트 합니다. 
Cloud환경에서 먼저 학습을 돌리기전에 테스트하는 과정이라고 생각하면 됩니다.

```bash
gcloud ml-engine local train \
  --job-dir $JOB_DIR \
  --module-name trainer.train \
  --package-path ./trainer
```

Cloud 환경에서 실제 학습을 다음과 같이 시킬수 있습니다.

````bash
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --runtime-version 1.8 \
  --module-name trainer.train \
  --package-path ./trainer \
  --region $REGION \
  --config config.yaml
````





