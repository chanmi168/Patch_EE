#!/bin/bash
 
# Declare an array of string with type
# declare -a TF_types=('target')
# declare -a variant=('baseline')
# declare -a input_names_list=('ECG_SR+PPG')

TF_type='target'
# variant='baseline'
input_names='ECG_SR+PPG_select'
# declare -a variants=('baseline' 'Late_UNet')
# declare -a variants=('Attention_UNet' 'Late_UNet' 'AT_bock' 'baseline')
# declare -a variants=('AT_block')
declare -a variants=('Attention_UNet' 'Late_UNet' 'AT_block' 'baseline')

inputdir='../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'

training_params_file='training_params_baseline_default.json'

# Iterate the string array using for loop
for variant in ${variants[@]}; do

    echo "===================== running stage 4 [${variant}] ====================="

    mkdir -p $outputdir
    python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file --input_names $input_names | tee $outputdir/training_logs_$variant.txt
    echo '============================================= DONE  ============================================='

done

