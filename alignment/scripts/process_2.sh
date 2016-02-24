#!/bin/bash

ITER=$1

python run_experiment_2.py -method 0 -iter ${ITER}
python run_experiment_2.py -method 1 -iter ${ITER}
python run_experiment_2.py -method 3 -iter ${ITER}
