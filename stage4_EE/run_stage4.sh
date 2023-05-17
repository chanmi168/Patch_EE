#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../data/stage3_norun/"
outputdir="../../data/stage4/"
training_params_file="training_params_baseline.json"

echo "===================================== running stage 4 [regression]  ====================================="
mkdir -p $outputdir
python DL_regression.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
