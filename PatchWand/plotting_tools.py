import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
# matplotlib.rc( 'savefig', facecolor = 'white' )
# matplotlib.rc( 'savefig', facecolor = 'black' )
from matplotlib.gridspec import GridSpec

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
pd.set_option('display.max_columns', 500)

import random
from random import randint
random.seed(0)

from setting import *

# import wandb



'''
Restricted Cubic Splines
For Pandas/Python
See https://apwheele.github.io/MathPosts/Splines.html
for class notes on how restricted cubic splines
are calculated
Andy Wheeler
'''

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import patsy
from scipy.stats import beta



i_seed = 0


# plt.style.use('dark_background')
confidence_interv = 90

color_dict = {'Red': '#e6194b', 
              'Green': '#3cb44b', 
              'Yellow': '#ffe119', 
              'Blue': '#0082c8', 
              'Orange': '#f58231', 
              'Purple': '#911eb4', 
              'Cyan': '#46f0f0', 
              'Magenta': '#e6194b',              
              'Navy': '#000080', 
              'Teal': '#008080', 
              'Brown': '#aa6e28', 
              'Maroon': '#800000', 
              'ForestGreen': '#228b22',
              'SteelBlue': '#4682B4',
              'MidnightBlue': '#1A4876',
              'RoyalPurple': '#7851A9',
              'MangoTando': '#FF8243',
              'Sunglow': '#FFCF48',
              'Lavender': '#e6beff', 
              'Lime': '#d2f53c', 
              'Pink': '#fabebe', 
              'Olive': '#808000', 
              'Coral': '#ffd8b1',
              'Cardinal': '#CC2336',
              'Black': '#000000',
              'Deep Carrot Orange': '#E4682A',
              'burntumber': '#8A3324',
              'darkgoldenrod': '#b8860b',
              'gold': '#FFD700',
              'Navy': '#000080',
              'Firebrick': '#b22222',
              'White': '#FFFFFF',
             }

color_names = ['Red',
 'Green',
 'Yellow',
 'Blue',
 'Orange',
 'Purple',
 'Cyan',
 'Magenta',
 'Navy',
 'Teal',
 'Brown',
 'Maroon',
 'ForestGreen',
 'SteelBlue',
 'MidnightBlue',
 'RoyalPurple',
 'MangoTando',
 'Sunglow',
 'Lavender',
 'Lime',
 'Pink',
 'Olive',
 'Coral',
 'Cardinal',
 'Black',
 'Deep Carrot Orange',
 'burntumber',
 'Firebrick']

sync_color_dict = {

    
    
    # patch
    'ECG': 'Blue',
    'ecg_beats': 'Blue',
    
    'accelX': 'MangoTando',
    'scg_x': 'MangoTando',
    'accelY': 'Olive',
    'scg_y': 'Olive',
    'accelZ': 'Teal',
    'scg_z': 'Teal',
    'SCG': 'Teal',

    'ppg_r_1': 'Maroon',
    'ppg_g_1': 'ForestGreen',
    'ppg_ir_1': 'darkgoldenrod',
    
    'ppg_r_2': 'Red',
    'ppg_g_2': 'Green',
    'ppg_ir_2': 'gold',
    'PPG': 'Maroon',
    
    'temp_skin': 'Pink',
    'pres': 'Deep Carrot Orange',
    
    # COSMED
    'HR_cosmed': 'Cyan',
    'RR_cosmed': 'Magenta',
    'VT_cosmed': 'Navy',
    'VE_cosmed': 'burntumber', # VE = VT x RR
    
    'VO2_cosmed': 'Cardinal',
    'VCO2_cosmed': 'Coral',
    'EE_cosmed': 'RoyalPurple',
    'SPO2_cosmed': 'Firebrick',
    
    'OUES_cosmed': 'MidnightBlue',
    
}


random_colors = []
for i in range(60):
    random_colors.append('#%06X' % randint(0, 0xFFFFFF))
    

task_palette = {
    0: '#e6194b',
    1: '#3cb44b',
    2: '#ffe119',
    3: '#0082c8',
    4: '#f58231',
    5: '#911eb4',
    6: '#46f0f0',
    7: '#e6194b',
    8: '#000080',
    9: '#008080',
    10: '#aa6e28',
    11: '#800000',
}

subject_palette = {
    0: '#e6194b',
    1: '#3cb44b',
    2: '#ffe119',
    3: '#0082c8',
    4: '#f58231',
    5: '#911eb4',
    6: '#46f0f0',
    7: '#e6194b',
    8: '#000080',
    9: '#008080',
    10: '#aa6e28',
    11: '#800000',
    12: '#228b22',
    13: '#4682B4',
    14: '#1A4876',
    15: '#7851A9',
    16: '#FF8243',
    17: '#FFCF48',
    18: '#e6beff',
    19: '#d2f53c',
    20: '#fabebe',
    21: '#808000',
    22: '#ffd8b1',
    23: '#CC2336',
    24: '#000000',
    25: '#E4682A',
    26: '#8A3324',
    
    100: '#e6194b',
    101: '#3cb44b',
    102: '#ffe119',
    103: '#0082c8',
    104: '#f58231',
    105: '#911eb4',
    106: '#46f0f0',
    107: '#e6194b',
    108: '#000080',
    109: '#008080',
    110: '#aa6e28',
    111: '#800000',
    112: '#228b22',
    113: '#4682B4',
    114: '#1A4876',
    115: '#7851A9',
    116: '#FF8243',
    117: '#FFCF48',
    118: '#e6beff',
    119: '#d2f53c',
    120: '#fabebe',
    121: '#808000',
    122: '#ffd8b1',
    123: '#CC2336',
    124: '#000000',
    125: '#E4682A',
    126: '#8A3324',
    212: '#228b22',
}

subject_str_palette = {
    '100': '#e6194b',
    '101': '#3cb44b',
    '102': '#ffe119',
    '103': '#0082c8',
    '104': '#f58231',
    '105': '#911eb4',
    '106': '#46f0f0',
    '107': '#e6194b',
    '108': '#000080',
    '109': '#008080',
    '110': '#aa6e28',
    '111': '#800000',
    '112': '#228b22',
    '113': '#4682B4',
    '114': '#1A4876',
    '115': '#7851A9',
    '116': '#FF8243',
    '117': '#FFCF48',
    '118': '#e6beff',
    '119': '#d2f53c',
    '120': '#fabebe',
    '121': '#808000',
    '122': '#ffd8b1',
    '123': '#CC2336',
    '124': '#000000',
    '125': '#E4682A',
    '126': '#8A3324',
    '212': '#228b22',
}

# sig_color_dict = {
#     'ECG_biopac': 'Cyan',
#     'PPG_biopac': 'Magenta',
#     'spiro_biopac': 'Navy',
#     'SpO2_biopac': 'Firebrick',
    
    
#     # patch
#     'ECG_filt': 'Blue',
#     'ECG': 'Blue',
    
#     'accelX': 'MangoTando',
#     'accelY': 'Olive',
#     'accelZ': 'Teal',
#     'accelX_filt': 'MangoTando',
#     'accelY_filt': 'Olive',
#     'accelZ_filt': 'Teal',

#     'ppg_r_1': 'Maroon',
#     'ppg_g_1': 'ForestGreen',
#     'ppg_ir_1': 'darkgoldenrod',
#     'ppg_r_1_filt': 'Maroon',
#     'ppg_g_1_filt': 'ForestGreen',
#     'ppg_ir_1_filt': 'darkgoldenrod',
    
#     'ppg_r_2': 'Red',
#     'ppg_g_2': 'Green',
#     'ppg_ir_2': 'gold',
#     'ppg_r_2_filt': 'Red',
#     'ppg_g_2_filt': 'Green',
#     'ppg_ir_2_filt': 'gold',
    
    
# #     'accelZ_filt': 'Teal',
# #     'accelY_filt': 'RoyalPurple', # caudal-cranial (C-C)
# #     'accelX_filt': 'Coral', # left-right (L-R),
# }
# plotted_sigs = ['ECG_biopac', 'PPG_biopac', 'spiro_biopac', 'SpO2_biopac', 'ECG_filt', 'accelZ_filt', 'ppg_r_1_filt', 'ppg_ir_1_filt', 'ppg_r_2_filt', 'ppg_ir_2_filt']

# sig_color_dict = {
#     'ECG': 'Blue',
    
#     'ppg_r': 'Red',
#     'ppg_g': 'Green',
#     'ppg_ir': 'Orange',
    
# #     'ppg_r_2': 'Maroon',
# #     'ppg_g_2': 'ForestGreen',
# #     'ppg_ir_2': 'Brown',
    
#     'SCG': 'Teal',
# #     'SCG-CC': 'RoyalPurple', # caudal-cranial (C-C)
# #     'SCG-LR': 'Coral', # left-right (L-R),
# }

marker_dict = {
    'circle': 'o',
    'triangle_down': 'v',
    'tri_down': '1',
    'square': 's',
    'pentagon': 'p',
    'plus': 'P',
    'star': '*',
    'hexagon2': 'H',
    'x': 'X',
    'diamond': 'D',
    

    'triangle_up': '^',
    'triangle_right': '>',
    'tri_up': '2',
    'octagon': '8',
    'hexagon1': 'h',
    'x (filled)': 'X',
    'thin_diamond': 'd',
    'alpha': r'$\alpha$',
    'music': r'$\u266B$',
    'lambda': r'$\lambda$',

}

input_marker_dict = {
    'ECG_filt': 'o',
    'scgZ': 'X',
    'ppg_ir_2_cardiac': 'v',
    'ppg_g_2_cardiac': 'v',
    'ppg_r_2_cardiac': 'v',
    'ppg_ir_1_cardiac': 'v',
    'ppg_g_1_cardiac': 'v',
    'ppg_r_1_cardiac': 'v',
}

input_color_dict = {
    'ECG_filt': '#0082c8',
    'scgX': color_dict['MangoTando'],
    'scgY': color_dict['Olive'],
    'scgZ': 'Teal',
    'SCG_merged': 'Teal',
       
    'ECG': '#0082c8',
    'SCG': 'Teal',
    'PPG': 'Maroon',
    
    'ppg_r_1_cardiac': 'Maroon',
    'ppg_g_1_cardiac': 'ForestGreen',
    'ppg_ir_1_cardiac': 'darkgoldenrod',
    
    'ppg_r_2_cardiac': 'Red',
    'ppg_g_2_cardiac': 'Green',
    'ppg_ir_2_cardiac': 'gold',
    
    'ppg_r_1_resp': color_dict['Maroon'],
    'ppg_g_1_resp': color_dict['ForestGreen'],
    'ppg_ir_1_resp': color_dict['darkgoldenrod'],
    
    'ppg_r_2_resp': color_dict['Red'],
    'ppg_g_2_resp': color_dict['Green'],
    'ppg_ir_2_resp': color_dict['gold'],

}



Fitz_dict = {
    1: '#F6D0B1',
    2: '#E8B58F',
    3: '#D29F7C',
    4: '#BC7951',
    5: '#A65E2B',
    6: '#3B1F1B',
}

def ax_no_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    
