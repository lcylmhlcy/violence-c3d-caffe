#!/bin/bash

LOG=examples/violence_c3d/Log/train-`date +%Y-%m-%d-%H-%M-%S`.log

./build/tools/caffe train --solver=examples/violence_c3d/c3d_solver.prototxt --gpu=0 2>&1 | tee $LOG