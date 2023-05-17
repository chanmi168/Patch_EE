#!/bin/bash

set -e
set -u



inputdir="../../data/stage1/"
outputdir="../../data/stage2/"
cosmed_unfiltered="True"
saving_format='csv'
# cosmed_unfiltered="False"
# saving_format='feather'

# for i in 1 2 3 4 5 6 7 8 9
# for i in 1 2 3 4 6 7 8 10
# for i in 11 12 13 14
# for i in 101 102 103 104 105 106 107 108 109 110 111 113 114 115 116 117 118 119 120 121 212
for i in 120 121 212
# for i in 212
    do
#         subject_id="sub$((i + 100))"
        subject_id="sub$((i))"
        echo 


        echo "===================================== running stage 2 [$subject_id, synchronize patch and cosmed files]  ====================================="


        # subject_id='sub101'

        mkdir -p $outputdir
        python sync.py --input_folder $inputdir --output_folder $outputdir --subject_id $subject_id --cosmed_unfiltered $cosmed_unfiltered --saving_format $saving_format | tee $outputdir/training_logs_$subject_id.txt
        echo "===================================== DONE ====================================="


    done



# #!/bin/bash

# set -e
# set -u

# echo '===================================== running stage 2 [synchronize patch and cosmed files]  ====================================='


# # args = parser.parse_args(['--input_folder', '../../data/stage1/', 
# #                           '--output_folder', '../../data/stage2/',
# #                           '--subject_id', 'sub101',
                          
                          
# inputdir='../../data/stage1/'
# outputdir='../../data/stage2/'


# for (( c=1; c<=5; c++ ))
# do  
#    echo "Welcome $c times"
# done
# # subject_id='sub101'

# # mkdir -p $outputdir
# # python sync.py --input_folder $inputdir --output_folder $outputdir --subject_id $subject_id
# echo '===================================== DONE ====================================='

