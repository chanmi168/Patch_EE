#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../data/stage3/win60_overlap90/"
outputdir="../../data/stage4_FL/"

echo "===================================== running stage 4 [HR estimate baseline]  ====================================="
training_params_file="training_params_baseline.json"

mkdir -p $outputdir
python feature_learning.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="

# echo "===================================== running stage 4 [HR estimate 6MWT recovery]  ====================================="
# training_params_file="training_params_6MWTrecovery.json"

# mkdir -p $outputdir
# python feature_learning.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
# echo "===================================== DONE ====================================="



# echo "===================================== running stage 4 [HR estimate 6MWT]  ====================================="
# training_params_file="training_params_6MWT.json"

# mkdir -p $outputdir
# python feature_learning.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
# echo "===================================== DONE ====================================="
