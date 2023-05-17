#!/bin/bash
 
# Declare an array of string with type
# declare -a TF_types=('target' 'source')
declare -a TF_types=('target')
# declare -a variants=('SE_block' 'baseline')
declare -a variants=('AT_block')
# declare -a variants=('AT_block' 'baseline')
# declare -a variants=('baseline')

# declare -a input_names_list=('ECG_SR' 'SCGxyz_AMpt' 'SCG_AMpt' 'SCGx_AMpt' 'SCGy_AMpt' 'SCG_select' 'accelX_resp' 'accelY_resp' 'accelZ_resp' 'ppg_g_1_resp' 'ppg_g_2_resp' 'ppg_ir_1_resp' 'ppg_ir_2_resp' 'PPG' 'PPG_select' 'ECG_SR+SCG_AMpt' 'ECG_SR+PPG' 'SCG_AMpt+PPG' 'ECG_SR+SCG_AMpt+PPG')
declare -a input_names_list=('ECG_SR' 'SCG_select' 'PPG_select' 'ECG_SR+SCG_select' 'ECG_SR+PPG_select' 'SCG_select+PPG_select' 'ECG_SR+SCG_select+PPG_select')

# declare -a input_names_list=('SCGxyz_AMpt' 'SCG_AMpt' 'SCGx_AMpt' 'SCGy_AMpt' 'accelX_resp' 'accelY_resp' 'accelZ_resp' 'ppg_g_1_resp' 'ppg_g_2_resp' 'ppg_ir_1_resp' 'ppg_ir_2_resp')

# declare -a input_names_list=('ECG_SR+SCG_select+PPG_select' 'SCG_select')

# declare -a input_names_list=('ECG_SR' 'PPG')




 
inputdir='../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'
training_params_file='training_params_baseline_default.json'


# Iterate the string array using for loop
for variant in ${variants[@]}; do
    for TF_type in ${TF_types[@]}; do
        for input_names in ${input_names_list[@]}; do
            echo "===================== running stage 4 [$input_names] ====================="

            mkdir -p $outputdir
            python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file --input_names $input_names | tee $outputdir/training_logs_$TF_type.txt
            echo '============================================= DONE  ============================================='

        done
    done
done