#!/bin/bash

export BUCKET_NAME=anderson-mnist
export JOB_NAME="mnist_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --runtime-version 1.8 \
  --module-name trainer.task \
  --package-path ./trainer \
  --region $REGION \
  --config trainer/config.yaml \
  -- \
  --train-file gs://$BUCKET_NAME

