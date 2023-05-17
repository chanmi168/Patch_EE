#!/bin/bash
 
# Declare an array of string with type

# declare -a TF_types=('prepare' 'target')
declare -a TF_types=('prepare')
# declare -a variant=('baseline' 'SE_block')
# declare -a variant=('AT_block')
declare -a variant=('AT_block')
input_names='ECG_SR+ECG_SR'


inputdir='../../data/stage3-1_windowing/GT_dataset/win60_overlap95_seq20_norm/'
outputdir='../../data/stage4_UNet/'
training_params_file='training_params_baseline_default.json'

# Iterate the string array using for loop
for variant in ${variant[@]}; do
    for TF_type in ${TF_types[@]}; do
        echo "===================== running stage 4 [${input_names}] ====================="
#         training_params_file='training_params_baseline_ECG.json'

        mkdir -p $outputdir
        python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file --input_names $input_names | tee $outputdir/training_logs_$TF_type.txt
        echo '============================================= DONE  ============================================='
    done
done

#         echo '===================== running stage 4 [SCG] ====================='
#         training_params_file='training_params_baseline_SCG.json'

#         mkdir -p $outputdir
#         python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --variant $variant --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#         echo '============================================= DONE  ============================================='

#         echo '===================== running stage 4 [ECGSCG] ====================='
#         training_params_file='training_params_baseline_ECGSCG.json'

#         mkdir -p $outputdir
#         python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#         echo '============================================= DONE  ============================================='

#     done
# done


# inputdir='../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/'
# outputdir='../../data/stage4_UNet/'

# # Iterate the string array using for loop
# for TF_type in ${TF_types[@]}; do

#     echo '===================== running stage 4 [ECG] ====================='
#     training_params_file='training_params_baseline_ECG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [PPG] ====================='
#     training_params_file='training_params_baseline_PPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [SCG] ====================='
#     training_params_file='training_params_baseline_SCG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [ECGPPG] ====================='
#     training_params_file='training_params_baseline_ECGPPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [ECGSCG] ====================='
#     training_params_file='training_params_baseline_ECGSCG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [SCGPPG] ====================='
#     training_params_file='training_params_baseline_SCGPPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [multiple modalities] ====================='
#     training_params_file='training_params_baseline_All.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [x11 modalities] ====================='
#     training_params_file='training_params_baseline_combo.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='

# done

