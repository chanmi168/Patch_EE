#!/bin/bash

set -e
set -u




inputdir='../../data/raw/'
outputdir='../../data/stage1/'
# filter_window='5'

# for i in 1 2 3 4 5 6 7
# for i in  1 2 3 4 5 6 7 8 9
# for i in 16
# for i in 1 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 20

# filter_window='5'
# for i in 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 212


filter_window='1'
for i in 120 121 212

# for i in 212
# for i in 5


    do
        # subject_id='sub107'
        subject_id="sub$((i))"

        echo "===================================== running stage 1 [$subject_id, parse cosmed data]  ====================================="

        mkdir -p $outputdir
        python read_cosmed.py  --input_folder $inputdir --output_folder $outputdir --subject_id $subject_id --filter_window $filter_window | tee $outputdir/logs_cosmed.txt
        echo '===================================== DONE ====================================='


#         echo "===================================== running stage 1 [$subject_id, parse patch data]  ====================================="

# mkdir -p $outputdir
#         python read_patch.py  --input_folder $inputdir --output_folder $outputdir --subject_id $subject_id | tee $outputdir/logs_patch.txt
#         echo '===================================== DONE ====================================='

    done