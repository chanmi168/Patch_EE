#!/bin/bash
 
# Declare an array of string with type
# declare -a TF_types=('source' 'FT_top' 'FT_top2' 'FT_all' 'target' )
# inputdir='../../data/stage3-1_windowing/win60_overlap95_seq20_norm/'
# declare -a TF_types=('pretrain')

TF_type='pretrain'

outputdir='../../data/stage4_UNet/'

declare -a inputdirs=('../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/' '../../data/stage3-1_windowing/GT_dataset/win60_overlap95_seq20_norm/')
# declare -a inputdirs=('../../data/stage3-1_windowing/GT_dataset/win60_overlap95_seq20_norm/')

# Iterate the string array using for loop
for inputdir in ${inputdirs[@]}; do

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



#     echo '===================== running stage 4 [PPG] ====================='
#     training_params_file='training_params_baseline_PPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


#     echo '===================== running stage 4 [ECGPPG] ====================='
#     training_params_file='training_params_baseline_ECGPPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='


# echo '===================== running stage 4 [SCGPPG] ====================='
#     training_params_file='training_params_baseline_SCGPPG.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='

#     echo '===================== running stage 4 [multiple modalities] ====================='
#     training_params_file='training_params_baseline_All.json'

#     mkdir -p $outputdir
#     python unet_TEST.py  --input_folder $inputdir --output_folder $outputdir --TF_type $TF_type --training_params_file $training_params_file | tee $outputdir/training_logs_$TF_type.txt
#     echo '============================================= DONE  ============================================='




done