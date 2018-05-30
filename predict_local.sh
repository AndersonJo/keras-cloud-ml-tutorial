#!/bin/bash

gcloud ml-engine local predict \
  --model-dir checkpoints/model \
  --json-instances sample.json \
  --verbosity debug
