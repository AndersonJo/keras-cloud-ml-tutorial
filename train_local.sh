#!/bin/bash

gcloud ml-engine local train \
  --job-dir checkpoints \
  --module-name trainer.task \
  --package-path ./trainer \
  -- \
  --train-file gs://anderson-mnist

