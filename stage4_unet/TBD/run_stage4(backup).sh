#!/bin/bash

set -e
set -u

inputdir='../../data/stage3-1_windowing/win60_overlap95_seq20_norm/'


# TF_type='source'

echo '===================== running stage 4 [multiple modalities] ====================='
training_params_file='training_params_baseline_All.json'
outputdir='../../data/stage4_UNet/All/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [ECG] ====================='
training_params_file='training_params_baseline_ECG.json'
outputdir='../../data/stage4_UNet/ECG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [PPG] ====================='
training_params_file='training_params_baseline_PPG.json'
outputdir='../../data/stage4_UNet/PPG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [SCG] ====================='
training_params_file='training_params_baseline_SCG.json'
outputdir='../../data/stage4_UNet/SCG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [ECGPPG] ====================='
training_params_file='training_params_baseline_ECGPPG.json'
outputdir='../../data/stage4_UNet/ECGPPG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [ECGSCG] ====================='
training_params_file='training_params_baseline_ECGSCG.json'
outputdir='../../data/stage4_UNet/ECGSCG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='


echo '===================== running stage 4 [SCGPPG] ====================='
training_params_file='training_params_baseline_SCGPPG.json'
outputdir='../../data/stage4_UNet/SCGPPG/'

mkdir -p $outputdir
python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file | tee $outputdir/training_logs.txt
echo '============================================= DONE  ============================================='
