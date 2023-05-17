#!/bin/bash

set -e
set -u



inputdir="../../data/stage2/"
outputdir="../../data/stage3/"
training_params_file="training_params_resp.json"

echo "===================================== running stage 3 [proprocess all files]  ====================================="
mkdir -p $outputdir
python preprocess.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file
echo "===================================== DONE ====================================="
