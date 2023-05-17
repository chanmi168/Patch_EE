#!/bin/bash
 
# Declare an array of string with type
declare -a TF_types=('target' 'source')
# declare -a TF_types=('target')
# declare -a variant=('SE_block' 'baseline')
# declare -a variant=('AT_block')
declare -a variant=('AT_block' 'baseline')

inputdir='../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'



# Iterate the string array using for loop
for variant in ${variant[@]}; do
    for TF_type in ${TF_types[@]}; do

        if [ $variant != 'AT_block' ]; then


        echo '===================== running stage 4 [ECG] ====================='
        training_params_file='training_params_baseline_ECG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


        echo '===================== running stage 4 [PPG] ====================='
        training_params_file='training_params_baseline_PPG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


        echo '===================== running stage 4 [SCG] ====================='
        training_params_file='training_params_baseline_SCG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='

        else
        echo "ignore single modality for AT_block"
        fi


        echo '===================== running stage 4 [ECGPPG] ====================='
        training_params_file='training_params_baseline_ECGPPG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


        echo '===================== running stage 4 [ECGSCG] ====================='
        training_params_file='training_params_baseline_ECGSCG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


        echo '===================== running stage 4 [SCGPPG] ====================='
        training_params_file='training_params_baseline_SCGPPG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


        echo '===================== running stage 4 [multiple modalities] ====================='
        training_params_file='training_params_baseline_All.json'
    #       "input_names": ["ECG_SR", "SCG_AMpt", "ppg_g_2_resp", "ppg_ir_2_resp"],

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='


    #     echo '===================== running stage 4 [x11 modalities] ====================='
    #     training_params_file='training_params_baseline_combo.json'

    #     mkdir -p $outputdir
    #     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
    #     echo '============================================= DONE  ============================================='

    done
done