#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../../data/stage3/win60_overlap90/"
outputdir="../../../data/stage4/ML_regression/exp_features/"
training_params_file="training_params_ML.json"

echo "===================================== running stage 4 [VO2 estimate ML]  ====================================="

mkdir -p $outputdir
python ML_regression.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
