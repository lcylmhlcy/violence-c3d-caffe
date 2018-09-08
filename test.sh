#!/usr/bin/env bash

# model architecture
MODELDEF=examples/violence_c3d/c3d_train_test.prototxt
LASTMODEL=examples/violence_c3d/violence_c3d_iter_1000.caffemodel

build/tools/caffe test --model=${MODELDEF} --weights=${LASTMODEL}  --gpu=0 

