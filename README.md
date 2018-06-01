[TOC]

# Overview

Keras를 Google Cloud ML Engine에서 학습시키고 서빙까지 하는 방법을 튜토리얼로 제공합니다. 



# Structure

Google Cloud ML Engine에서 사용하기 위해서는 다음과 같은 **최소한의** 구조를 갖고 있어야 합니다. 

```
.
├── config.yaml
├── README.md
├── setup.py
├── trainer
│   ├── __init__.py
│   ├── config.yaml
    └── train.py
```



# Virtual Environment

먼저 virtual environment 를 생성합니다. 

> 만약 Python2.7을 기본으로 사용하면 굳이 해줄필요는 없습니다. 
> gcloud ml-engine 명령어가 python 2.7 을 기본으로 사용하고 있기 때문입니다. 

```bash
virtualenv cmle --python=/usr/local/bin/python3.6 --system-site-packages
source cmle/bin/activate
```



# Google Cloud SDK

[Google Cloud SDK 설치방법](https://github.com/AndersonJo/google-cloud-platform/blob/master/01-quickstart.md) 을 참고하여 설치를 합니다. 

설치후 다음과 같은 명령어를 사용할 수 있습니다. 

**모델 리스트**

```bash
gcloud ml-engine models list
```



# MNIST Dataset

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



# Model 생성및 TensorFlow Moldel로 저장 함수

```python
from typing import Tuple, List

import keras.backend as K
import tensorflow as tf
from keras import Sequential, Model, Input
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from tensorflow.python.saved_model import builder as tf_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.training.momentum import MomentumOptimizer


# Model
def create_model() -> Tuple[Model, tf.Tensor]:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    image_input = Input(shape=(784,))

    h = Dense(512, activation='relu', input_shape=(784,))(image_input)
    h = Dropout(0.2, seed=0)(h)
    h = Dense(256, activation='relu')(h)
    h = Dropout(0.2, seed=0)(h)
    h = Dense(10)(h)
    output = Activation('softmax')(h)
    arg_max = K.argmax(output)

    model = Model(image_input, [output])
    model.compile(loss='categorical_crossentropy',
                  optimizer=MomentumOptimizer(0.01, momentum=0.9),
                  metrics=['accuracy'])

    return model, arg_max
```

Keras Model을 그냥 저장시키면 안되고 TensorFlow Model 형태로 저장을 해야 합니다. 

아래의 함수를 사용해서 TensorFlow Model로 저장할수 있습니다. 

```python
# Save Model
def save_as_tensorflow(model: Model, export_path: str, arg_max: tf.Tensor):
    builder = tf_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'image': model.inputs[0]},
                                      outputs={'probabilities': model.outputs[0],
                                               'class': arg_max})
    sess = K.get_session()
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()
```





# Trainer 

trainer 파이썬 패키지안에 task.py를 생성합니다. 

```python
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
    train_x, train_y, test_x, test_y = load_mnist(parser.train_file)
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





# config.yaml 설정

Cloud ML Engine내에서 학습시킬때 Python 3.5를 사용하게 하거나 (default는 python2.7), GPU를 사용하기 위해서는 `config.yaml` 같은 파일을 만들고 설정파일을 넣습니다. 

먼저 config.yaml을 trainer 디렉토리안에 생성을 합니다.

```
cd trainer
vi config.yaml
```

설정은 다음과 같이 합니다. 

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



## Python 3.5 사용

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



## GPU 설정

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




## Machine Type 설정

[Machine Types](https://cloud.google.com/ml-engine/docs/tensorflow/machine-types) 을 참고 합니다. 




# setup.py 설정하기

아래와 같이 설정을 합니다.

```python
from setuptools import setup, find_packages

setup(
    name='keras-cloud-ml-engine-tutorial',
    version='0.1',
    packages=find_packages(),  # ['trainer'],
    url='',
    license='MIT',
    author='anderson',
    author_email='a141890@gmail.com',
    description='',
    install_requires=[
        'keras',
        'h5py',
        'six'
    ]
)
```

실제 cloud ml engine상에서 돌려보면 필요한 라이브러리를 설치한 후에 실행이 됩니다. 
이때 참고 되는 파일이 setup.py입니다.



# gcloud 사용해서 학습시키기

먼저 변수들을 선언해줍니다.

```bash
export BUCKET_NAME=anderson-mnist
export JOB_NAME="mnist_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$CHECKPOINT
export REGION=us-east1
```

## Local 환경에서 학습

Local에서 학습을 테스트 합니다. 
Cloud환경에서 먼저 학습을 돌리기전에 테스트하는 과정이라고 생각하면 됩니다.

```bash
gcloud ml-engine local train \
  --job-dir checkpoints \
  --module-name trainer.task \
  --package-path ./trainer \
  -- \
  --train-file gs://$BUCKET_NAME
```

## Cloud 환경에서 학습

Cloud 환경에서 실제 학습을 다음과 같이 시킬수 있습니다.

````bash
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --runtime-version 1.8 \
  --module-name trainer.task \
  --package-path ./trainer \
  --region $REGION \
  --config trainer/config.yaml \
  -- \
  --train-file gs://$BUCKET_NAME
````

- **JOB DIR**: 학습 뒤 결과가 저장되는 곳이며, Cloud Storage의 주소를 적으면 됩니다. 



# Local 환경에서 Predict

```bash
gcloud ml-engine local predict \
  --model-dir checkpoints/model \
  --json-instances sample.json \
  --verbosity debug
```



# Cloud ML에 Deploy하기

클라우드상에 설치된 모델을 ls 명령어로 확인을 합니다. 

```bash
gsutil ls gs://anderson-mnist/mnist_train_*
```

```
gs://anderson-mnist/mnist_train_20180530_144908/:
gs://anderson-mnist/mnist_train_20180530_144908/model/
gs://anderson-mnist/mnist_train_20180530_144908/packages/

gs://anderson-mnist/mnist_train_20180530_162809/:
gs://anderson-mnist/mnist_train_20180530_162809/model/
gs://anderson-mnist/mnist_train_20180530_162809/packages/
```

모델의 주소가 `gs://anderson-mnist/mnist_train_20180530_162809/model/` 라는 것을 확인했고 디플로이를 합니다. 

```bash
MODEL_BINARIES=gs://anderson-mnist/mnist_train_20180530_162809/model/
MODEL_NAME=mnist

gcloud ml-engine versions create v1 \
  --model $MODEL_NAME \
  --origin $MODEL_BINARIES \
  --runtime-version 1.8
```

클라우드상에 생성된 모델을 확인합니다.

```bash
gcloud ml-engine models list
```

```
NAME   DEFAULT_VERSION_NAME
mnist  v1
```

최종적으로 **예측**도 해봅니다.

```
gcloud ml-engine predict \
  --model $MODEL_NAME \
  --version v1 \
  --json-instances sample.json
```

```
CLASS  PROBABILITIES
0      [0.9993340373039246, ..., 0.0002702845085877925]
1      [1.169654979094048e-06, ..., 8.002129470696673e-05]
2      [2.6620080461725593e-05, ..., 8.218656262215518e-07]
3      [2.932660026999656e-05, ..., 0.00019038221216760576]
4      [5.971842398366789e-08, ..., 0.04169595614075661]
5      [0.0009943349286913872, ..., 0.0004937952617183328]
6      [5.941236668149941e-05, ..., 1.3123836879458395e-06]
7      [2.825109504556167e-06, ..., 0.004673224873840809]
8      [0.0004219221300445497, ..., 0.001314960652962327]
9      [0.000538919004611671, ..., 0.8796722888946533]
```

