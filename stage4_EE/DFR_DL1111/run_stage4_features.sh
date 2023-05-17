#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../../data/stage3/win60_overlap90/"
outputdir="../../../data/stage4/DFR_DL1111/exp_features/"


# will take 3.75 * 4 hrs = 15 hrs in total
# cuda 0
training_params_file="training_params_features.json"

echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="

mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
