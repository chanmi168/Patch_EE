#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../../data/stage3/win60_overlap90/"
outputdir="../../../data/stage4/DFR_DL1111/exp_weights/"

# will sweep through different auxillary choices (each weight takes 3.75hr)
# 3.75*11 = 41.25
# auxillary weights: 0, 0.001,0.01, 0.1,   0.2, 0.3, 0.4, 0.45, 0.495, 0.4995, 0.5
# main weight:       1, 0.8,  0.98, 0.998, 0.6, 0.4, 0.2, 0.1,  0.01,  0.001,  0


# 3.75*5 = 18.75
# auxillary weights: 0, 0.1, 0.2, 0.3, 0.4, 0.5
# main weight:       1, 0.8, 0.6, 0.4, 0.2, 0


# cuda = 1
# auxillary weights: 0.001,0.01, 0.1, 0.45, 0.495, 0.4995

training_params_file="training_params_weights2.json"

echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="

mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="
