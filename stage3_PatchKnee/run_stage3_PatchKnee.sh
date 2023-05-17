#!/bin/bash

set -e
set -u



inputdir="/mchan_Data/knee_dataset/"
outputdir="../../data/stage3_PatchKnee/"

for i in 102 106 107 108 109 110 111 113 114 115 116 117
# for i in 106 107 108
# for i in 109 110 111 113 114 115 116 117 118 119 120 121 212
# for i in 101 102 103 104 105 106 107 108 109 110 111 113 114 115 116 117 118 119 120 121 212

    do
        subject_id="$((i))"

        echo "===================================== running stage 3 link_PatchKnee [proprocess all files] subject $subject_id  ====================================="
        mkdir -p $outputdir
        python link_PatchKnee.py --input_folder $inputdir --output_folder $outputdir --subject_id $subject_id
        echo "===================================== DONE ====================================="

done