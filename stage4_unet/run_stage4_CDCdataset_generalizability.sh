#!/bin/bash
 
# Declare an array of string with type
# declare -a TF_types=('target' 'source')
declare -a TF_types=('source')
# declare -a variant=('SE_block' 'baseline')
declare -a variant=('AT_block')
input_names='ECG_SR+PPG_select'

inputdir='../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'
training_params_file='training_params_baseline_default.json'

# Iterate the string array using for loop
for variant in ${variant[@]}; do
    for TF_type in ${TF_types[@]}; do

        echo "===================== running stage 4 [${input_names}] ====================="
#         training_params_file='training_params_baseline_ECGPPG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file --input_names $input_names | tee $outputdir/training_logs_$TF_type_$variant.txt
        
        echo '============================================= DONE  ============================================='


    done
done