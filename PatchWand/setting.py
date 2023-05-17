list_positions = ['sternum', 'clavicle', 'ribcage']

# FS_tasks = ['Baseline 0', 'Standing 0', 'Proning 0', 'LL 0', 'LR 0', 'Cough 0', 'SpeakCasual 0', 'SpeakScripted 0', 
#               '6MWT 0', 'Recovery 0', '6MWT-R 0', 'Recovery 1', 'Stair 0', 'Recovery 2', 'Cough 1', 'SpeakCasual 1', 'SpeakScripted 1', 
#               'Walk 0', 'Recovery 3', 'Run 0', 'Recovery 4', 'Exhaustion 0', 'Recovery 5', 'Exhaustion 1', 'Recovery 6']

# updated on 10/2 based on Sungtae's request (added StairDown0, StairUp0, StairDown1)
FS_tasks = ['Baseline 0', 'Standing 0', 'Proning 0', 'LL 0', 'LR 0', 'Cough 0', 'SpeakCasual 0', 'SpeakScripted 0', 
              '6MWT 0', 'Recovery 0', '6MWT-R 0', 'Recovery 1', 'StairDown0', 'StairUp0', 'StairDown1', 'Recovery 2', 'Cough 1', 'SpeakCasual 1', 'SpeakScripted 1', 
              'Walk 0', 'Recovery 3', 'Run 0', 'Recovery 4', 'Exhaustion 0', 'Recovery 5', 'Exhaustion 1', 'Recovery 6']

tasks_dict = {
    'Baseline': 0, 
    'Recovery 6MWT': 1, 
    'Recovery 6MWT-R': 2, 
#     'Recovery Stair': 3, 
    'Recovery StairDown1': 3, 
    'Recovery Walk': 4,
    'Recovery Run': 5,
    '6MWT': 6,
    '6MWT-R': 7,
#     'Stair': 8,
    'Walk': 9,
    'Run': 10,
    'Recovery Treadmill': 101,
}
tasks_dict_reversed = {
    0: 'Baseline', 
    1: 'Recovery 6MWT',
    2: 'Recovery 6MWT-R',
#     3: 'Recovery Stair',
    3: 'Recovery StairDown1',
    4: 'Recovery Walk',
    5: 'Recovery Run',
    6: '6MWT',
    7: '6MWT-R',
#     8: 'Stair',
    9: 'Walk',
    10: 'Run',
    101: 'Recovery Treadmill'
}


# TODO: find correct range for VO2, VCO2, EE, VT, VE
label_range_dict = {    
    'width_QRS': 0.15, # sec
    'ppg': [-100, 100], # uW
    'acc': [-2, 2], # g
    'ecg': [-5, 5], # mV
    'RR': [5, 60], # breaths per minute
    'HR': [40, 220], # beats per minute
    'rer': [0.4, 1.0], # no unit since it's slope
    'vevco2_slope': [20, 50], # no unit since it's slope, where ve is in L/min, CO2 is in ml/min
#     'VT': [3, 40], # ml/kg
    'VT': [0, 500000], # ml/breath
    'SPO2': [5, 100], # %
    'VE': [0,5000000], # TODO: get ref, find unit
    'VO2': [0,5000000], # TODO: get ref, find unit
    'VCO2': [0,5000000], # TODO: get ref, find unit
    'EE': [0,100], # TODO: get ref, find unit
    'FiO2': [0,100], # TODO: get ref, find unit
    'FiCO2': [0,100], # TODO: get ref, find unit
    'FeO2': [0,100], # TODO: get ref, find unit
    'FeCO2': [0,100], # TODO: get ref, find unit
    'GpsAlt': [1,2000], # TODO: get ref, find unit
    'AmbTemp': [-10,35], # TODO: get ref, find unit
    'HR_DL': [40, 150],
}

unit_dict = {
#     'accel': 'g',
    'accel': 'mg', # please double check
    'ppg': 'uW',
    'temp': 'Celsius',
    'pres': 'mbar', # or hPa
    'ecg': 'mV',    
    'ECG': 'mV',    
    'PPG': 'uW',    
    'SCG': 'mg',    
    'HR': 'b(eats)pm',    
    'RR': 'b(reaths)pm',   
    'SPO2': '%',    

#     'VT': 'ml/kg/breath',    
#     'VE': 'ml/kg/min',    
#     'VO2': 'ml/kg/min',    
#     'VCO2': 'ml/kg/min',    
#     'EE': 'kcals/kg/min', 
    'VT': 'ml/breath',    
    'VE': 'ml/min',    
    'VO2': 'ml/min',    
    'VCO2': 'ml/min',    
    'EE': 'kcals/min', 
    'EErq': 'kcals/min', 
    'BMR': 'kcal/day',
    
    'FiO2': '%',  
    'FiCO2': '%',  
    'FeO2': '%',  
    'FeCO2': '%',  
    'OUES': 'a.u.',
    'O2pulse': 'ml/beat',
    
    'VT_cosmedperc': '%',
    'EErq_cosmedperc': '%',
#     'VT': 'ml/breath',    
#     'VE': 'ml/min',    
#     'VO2': 'ml/min',    
#     'VCO2': 'ml/min',    
#     'EE': 'kcals/min',    
}

# beat_unit_dict = {
#     'ecg(patch)': 'V',
#     'ecg(biopac)': 'mV',
#     'ppg_g_1(patch)': 'μA',
#     'ppg_r_1(patch)': 'μA',
#     'ppg_ir_1(patch)': 'μA',
#     'ppg_g_2(patch)': 'μA',
#     'ppg_r_2(patch)': 'μA',
#     'ppg_ir_2(patch)': 'μA',
#     'scg_z(patch)': 'g',
#     'ppg_r(biopac)': 'a.u.',
#     'i_R_peaks': None,
#     'SpO2_biopac': '%',
#     'ppg_g_1_DC': 'μA',
#     'ppg_r_1_DC': 'μA',
#     'ppg_ir_1_DC': 'μA',
#     'ppg_g_2_DC': 'μA',
#     'ppg_r_2_DC': 'μA',
#     'ppg_ir_2_DC': 'μA',
# }



label_names = ['br', 'heart_rate_cosmed', 'rer', 'vco2_ml_min', 've', 'vo2_ml_min', 'vt']
sig_names = ['ECG', 'accelX', 'accelY', 'accelZ']
# surrogate_names = ['ECG_AM', 'ECG_AMpt', 'ECG_SR', 'ECG_FM', 'ECG_BW', 'SCG_AM', 'SCG_AMpt', 'SCG_FM', 'SCG_BW', 'PEP_FM', 'ECG_SQI']
surrogate_names = ['ECG_AM', 'ECG_AMbi', 'ECG_AMr', 'ECG_AMs', 'ECG_AMpt', 'ECG_SR', 'ECG_FM', 'ECG_BW', 'SCG_AM', 'SCG_AMpt', 'SCG_BW', 'PEP_FM', 'ECG_SQI']
# list_good_surrogate = ['ECG_AM', 'ECG_AMpt', 'ECG_SR', 'SCG_AMpt', 'SCG_BW', 'PEP_FM', 'ECG_SQI']
list_good_surrogate = ['ECG_AM', 'ECG_AMbi', 'ECG_AMr', 'ECG_AMs', 'ECG_AMpt', 'ECG_SR', 'ECG_FM', 'ECG_BW', 'SCG_AMpt', 'SCG_BW', 'PEP_FM']


# width_QRS = 0.15 # sec
# FS_PPG = 1000
# FS_SCG = 2000
# FS_RESAMPLE = 100
FS_RESAMPLE = 250
FS_RESAMPLE_resp = 5 # Hz

# PPG_lowcutoff = .5
# SCG_lowcutoff = 10

R_highcutoff = 0.1

# FILT_PPG = [0.35, 4]
FILT_PPG = [1, 4]
# FILT_ECG = [10, 30]
FILT_ECG = [1, 30]
FILT_SCG = [1, 25]
# FILT_SCG = [5, 25]
FILT_RESP = [0.08, 1]

# anaerobic threhsold determined by RER
# ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3880081/pdf/jssm-04-29.pdf
RER_AT = 1.0

# smoothing_dur = 1 # second

# FILT_PPG = [0.1, 4]
# FILT_ECG = [10, 30]
# # FILT_ECG = [1, 30]
# # FILT_SCG = [1, 25]
# FILT_SCG = [5, 25]