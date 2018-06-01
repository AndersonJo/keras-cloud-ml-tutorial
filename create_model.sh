#!/bin/bash

MODEL_NAME=mnist
REGION=us-central1
gcloud ml-engine models create $MODEL_NAME --region=$REGION
