#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../../data/stage3/win60_overlap90/"
outputdir="../../../data/stage4/DFR_DL555/"
training_params_file="training_params_baseline.json"

echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="

mkdir -p $outputdir
python DFR_regression555.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
