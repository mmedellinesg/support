#!/bin/bash

python src/models/convert_support_training_data_to_feature_vectors.py \
    data/training-data.with-ids.tsv \
    data/training-data.features

python src/models/train_support_classifier.py data/training-data.features