Below is a list of experiments

| Experiment ID | command | dataset | TF_types | variant | data |
| --- | --- | --- | --- | --- | --- |
| 1 | bash run_stage4_preparing.sh | GT_dataset | 'prepare' | 'AT_block' | only ECG |
| 2 | bash run_stage4_GTdataset.sh | GT_dataset | 'target' | 'baseline' |  ECG & SCG |
| 3 | bash run_stage4_CDCdataset.sh | CDC_dataset | 'target' 'source' | 'AT_block' 'baseline' | everything |
| 4 | bash run_stage4_CDCdataset_source.sh | CDC_dataset | 'source' | 'AT_block' | only fusion |
| 5 | bash run_stage4_CDCdataset_selectalg.sh | CDC_dataset | 'target' | 'baseline' | each PPG or SCG channel, PPG or SCG select |
| 6 | bash run_stage4_CDCdataset_fusion.sh | CDC_dataset | 'target' | ??? | ECG+PPG |

TODO: replace all PPG/SCG with PPG/SCG_select in the json files