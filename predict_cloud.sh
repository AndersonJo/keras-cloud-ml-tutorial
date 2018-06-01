#!/bin/bash

MODEL_BINARIES=gs://anderson-mnist/mnist_train_20180530_162809/model/
MODEL_NAME=mnist

gcloud ml-engine versions create v1 \
  --model $MODEL_NAME \
  --origin $MODEL_BINARIES \
  --runtime-version 1.8
