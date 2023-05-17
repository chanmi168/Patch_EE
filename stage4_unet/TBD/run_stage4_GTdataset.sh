#!/bin/bash
 
# Declare an array of string with type
# declare -a TF_types=('source' 'FT_top' 'FT_top2' 'FT_all' 'target' )
# inputdir='../../data/stage3-1_windowing/win60_overlap95_seq20_norm/'
# declare -a TF_types=('pretrain' 'target' 'source')
# declare -a TF_types=('source' 'target')
declare -a TF_types=('target')
# declare -a TF_types=('source')
inputdir='../../data/stage3-1_windowing/GT_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'

# Iterate the string array using for loop
for TF_type in ${TF_types[@]}; do

    echo '===================== running stage 4 [ECG] ====================='
    training_params_file='training_params_baseline_ECG.json'

    mkdir -p $outputdir
    python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
    echo '============================================= DONE  ============================================='


    echo '===================== running stage 4 [SCG] ====================='
    training_params_file='training_params_baseline_SCG.json'

    mkdir -p $outputdir
    python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
    echo '============================================= DONE  ============================================='


    echo '===================== running stage 4 [ECGSCG] ====================='
    training_params_file='training_params_baseline_ECGSCG.json'

    mkdir -p $outputdir
    python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
    echo '============================================= DONE  ============================================='

done