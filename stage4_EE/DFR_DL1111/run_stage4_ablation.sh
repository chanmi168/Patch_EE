#!/bin/bash

set -e
set -u

# inputdir="../../data/stage3/"
inputdir="../../../data/stage3/win60_overlap90/"
# cuda 0
# will take 3.75 * 6 hrs = 22.5hrs in total

# outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_nothing/"
# training_params_file="training_params_nothing.json"
# echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
# mkdir -p $outputdir
# python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
# echo "===================================== DONE ====================================="


# outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_FT/"
# training_params_file="training_params_FT.json"
# echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
# mkdir -p $outputdir
# python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
# echo "===================================== DONE ====================================="


outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_MTL/"
training_params_file="training_params_MTL.json"
echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="



outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_FTMTL/"
training_params_file="training_params_FTMTL.json"
echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="


outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_FTMTLCA/"
training_params_file="training_params_FTMTLCA.json"
echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="


outputdir="../../../data/stage4/DFR_DL1111/ablation/exp_FTMTLCASA/"
training_params_file="training_params_FTMTLCASA.json"
echo "===================================== running stage 4 [VO2 estimate baseline]  ====================================="
mkdir -p $outputdir
python DFR_regression1111.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo "===================================== DONE ====================================="

