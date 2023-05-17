import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.ticker as plticker

# %matplotlib inline

import numpy as np
from filters import *
from setting import *
from plotting_tools import *
from segmentation import *
from PPG_module import *


PLOTTED_SIGS = ['ECG', 'accelX', 'accelY', 'accelZ', 'ppg_r_1',  'ppg_ir_1',  'ppg_g_1', 'ppg_r_2',  'ppg_ir_2',  'ppg_g_2', 'pres', 'temp_skin',
            'HR_cosmed', 'RR_cosmed', 'VT_cosmed', 'VE_cosmed', 'VO2_cosmed', 'VCO2_cosmed', 'EE_cosmed', 'SPO2_cosmed']

def inspect_labels(ax, df, y_min, y_max, annotate=True, fontsize=15):
    task_label = df['task'].values    

    ts = df['time'].values

    TASKS = np.unique(task_label)
#     print(TASKS)

    for i, task_name in enumerate(FS_tasks):
        if task_name not in TASKS:
            continue

        indices = np.where(task_label==task_name)[0]

        task_start = ts[indices[0]]
        task_end = ts[indices[-1]]

        task_id = FS_tasks.index(task_name)
#         print(task_name, task_start, task_end)
        ax.fill_between( np.array( [ task_start, task_end ] ),
                             y_min * np.array( [1, 1] ),
                             y_max * np.array( [1, 1] ),
                             facecolor = color_dict[color_names[task_id+1]],
                             alpha = 0.1) 
        if annotate:
            annotate_alpha = 0.8

            text = ax.annotate(task_name, (task_start/2+task_end/2, y_max), fontsize=fontsize, color='black', horizontalalignment='center', verticalalignment='bottom', rotation=45)
            text.set_alpha(annotate_alpha)
            
def plot_all_sync(df_sub, subject_id, plt_scale=1, plotted_sigs=None, fig_name=None, outputdir=None, show_plot=False):

    if plotted_sigs is None:
        plotted_sigs = PLOTTED_SIGS
    df = df_sub.copy()
#     subject_id = df['subject_id'].unique()[0]

    df['time'] = df['time']-df['time'].values[0]

    t_arr = df['time'].values
#     t_arr = t_arr - t_arr[0]

    t_start = t_arr[0]
    t_end = t_arr[-1]

    t_dur = t_arr[-1] - t_arr[0]

    fig, axes = plt.subplots(len(plotted_sigs), 1, figsize=(plt_scale*20, len(plotted_sigs)), gridspec_kw = {'wspace':0, 'hspace':0}, dpi=80)

    # TODO: make a plot dict
    fontsize = 20/plt_scale
    linewidth = 2
    alpha = 0.8

    for i, ax in enumerate(axes):
        # condition grid
        ax.grid('on', linestyle='--')
        # no x ticks except for the bottom ax
        if i<len(axes)-1:
            ax.set_xticklabels([])
        # add y ticks to all axes
        ax.tick_params(axis='y', which='both', labelsize=20)

        sig_name = plotted_sigs[i]
        sig_plt = df[sig_name].values

        if sig_name in sync_color_dict.keys():
            color = color_dict[sync_color_dict[sig_name]]
        else:
            color = random_colors[i]
        ax.plot(t_arr, sig_plt, color=color, alpha=alpha ,zorder=1, linewidth=linewidth)
        ax.set_xlim(t_start, t_end) # remove the weird white space at the beg and end of the plot


        # remove some borders (top and right)
        ax.spines['right'].set_visible(False)
        if i==0:
            ax.spines['top'].set_visible(False)

        # add y label, indicate their unit
        if 'ECG' in sig_name:
            sig_unit = unit_dict['ecg']   
        elif 'ppg' in sig_name:
            sig_unit = unit_dict['ppg']   
        elif 'accel' in sig_name:
            sig_unit = unit_dict['accel']  
        elif 'pres' in sig_name:
            sig_unit = unit_dict['pres']  
        elif 'temp_skin' in sig_name:
            sig_unit = unit_dict['temp']   
        elif 'cosmed' in sig_name:
            sig_unit = unit_dict[sig_name.split('_')[0]]
        else:
            sig_unit = 'a.u.'
            
        ax.set_ylabel(sig_name + '\n({})'.format(sig_unit), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=100)
            
        # set tick font size
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)

        # set a hard limit on the range of the signals
        ylim_hard = False
        if ylim_hard:
            if 'ECG' in sig_name:
                ax.set_ylim(-2, 2)    
            if 'ppg' in sig_name:
                ax.set_ylim(-500, 500)
            if 'accel' in sig_name:
                ax.set_ylim(-2, 2)
            if 'cosmed' in sig_name:
                ax.set_ylim(label_range_dict[sig_name.split('_')[0]])
        if 'OUES' in sig_name:
            ax.set_ylim(0, 0.1)

        # do this so there's no weird white space on top and bottom of each ax
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max)
        

        
        # add color to each segment of all signal to indicate the task
        inspect_labels(ax, df, y_min, y_max, annotate=i==0)
        
        loc = plticker.MultipleLocator(base=100) # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

    ax.set_xlabel('time (sec)', fontsize=fontsize)
    fig.subplots_adjust(wspace=0, hspace=0)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'All_sub{}'.format(subject_id)
        else:
            fig_name = fig_name + '_sub{}'.format(subject_id)

        fig.savefig(outputdir + fig_name,bbox_inches='tight', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')




def plot_ALL_beats(beats_dict, beats_id, subject_id, Fs, show_good=None, fig_name=None, outputdir=None, show_plot=False):

    t_beat = np.arange(beats_dict['ecg_beats'].shape[0])/Fs
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    fontsize = 20
    alpha = 0.03
#     ymin, ymax = -0.025, 0.025
    ymin = []
    ymax = []

    for beat_name, beats in beats_dict.items():
        if 'biopac' in beat_name:
            continue
        if beat_name == 'i_R_peaks':
            continue
        if 'DC' in beat_name:
            continue
        if 'ppg' in beat_name:
            ymin.append(np.mean(beats,axis=1).min())
            ymax.append(np.mean(beats,axis=1).max())

    ymin = np.min(np.asarray(ymin))
    ymax = np.max(np.asarray(ymax))

    
    for (beat_name, beat_i) in zip(beats_dict, beats_id):
        if beat_name == 'i_R_peaks':
            continue
        if 'DC' in beat_name:
            continue

        beats = beats_dict[beat_name]
            
        ax = fig.add_subplot(3, 4, beat_i)
        ax.set_title(beat_name+'\n', fontsize=fontsize)

        color = color_dict[sync_color_dict[beat_name]]
        
        if show_good is not None:
            if 'ppg' in beat_name:
                
                if beat_name[-1]=='1':
                    template = beats_dict['ppg_r_1'].mean(axis=1)
                elif beat_name[-1]=='2':
                    template = beats_dict['ppg_r_1'].mean(axis=1)

                mask_all, ol_rate = clean_PPG(beats, template, Fs)
                if show_good==False:
                    beats = beats[:, ~mask_all]
#                     print(beat_name, ol_rate)
                    ax.set_title(beat_name+'\nrejection_rate:{:.2f}'.format(ol_rate), fontsize=fontsize)
                else:
                    beats = beats[:, mask_all]
#                     print(beat_name, ol_rate)
                    ax.set_title(beat_name+'\nacception_rate:{:.2f}'.format(1-ol_rate), fontsize=fontsize)
                    

        
        ax.plot(t_beat, beats, color='gray', alpha=alpha)
        ax.plot(t_beat, np.mean(beats,axis=1), color=color, linewidth=3)

#         if 'ppg' in beat_name:
#             ax.set_ylim(label_range_dict['ppg'])
#         if 'ecg' in beat_name:
#             ax.set_ylim(label_range_dict['ecg'])
#         if 'acc' in beat_name or 'scg' in beat_name:
#             ax.set_ylim(label_range_dict['acc'])

        if 'ppg' in beat_name or 'ecg' in beat_name or 'acc' in beat_name:

            beats_mean = np.mean(beats,axis=1)
#             ymin = np.min(beats_mean)
#             ymax = np.max(beats_mean)
#             ax.set_ylim(ymin-(ymax-ymin)*0.1, ymax+(ymax-ymin)*0.1)

            ymin = beats_mean.mean() - beats_mean.std()*3
            ymax = beats_mean.mean() + beats_mean.std()*3
            ax.set_ylim(ymin, ymax)        
        if 'scg' in beat_name:
            beats_mean = np.mean(beats,axis=1)
            ymin = beats_mean.mean() - beats_mean.std()*10
            ymax = beats_mean.mean() + beats_mean.std()*10
            ax.set_ylim(ymin, ymax)


        ax.tick_params(axis='both', which='major', labelsize=13)

        if 'scg' in beat_name:
            ax.set_xlim(0,0.4)
        else:
            ax.set_xlim(0,1)

        
        
#         if 'patch' in beat_name:
        if 'ppg' in beat_name:
            ax.set_ylabel(unit_dict['ppg'], fontsize=fontsize-3)
        if 'ecg' in beat_name:
            ax.set_ylabel(unit_dict['ecg'], fontsize=fontsize-3)
        if 'scg' in beat_name:
            ax.set_ylabel(unit_dict['accel'], fontsize=fontsize-3)
#         if 'biopac' in beat_name:
#             if 'ppg' in beat_name:
#                 ax.set_ylabel('a.u.', fontsize=fontsize-3)
#             if 'ecg' in beat_name:
#                 ax.set_ylabel('mV', fontsize=fontsize-3)

        
    ax.set_xlabel('time (sec)', fontsize=fontsize)
    
    
    
    fig.tight_layout()

    if outputdir is not None:
        
        if fig_name is None:
            fig_name = 'beats_ensemble_sub{}'.format(subject_id)

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name+'.png', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        

#     if log_wandb:
#         wandb.log({fig_name: wandb.Image(fig)})


def get_ensemble_beats(sig_beats, N_enBeats=4, use_woody=False):
    # i_R_peaks = (N_beats,)
    # sig_beats = (N_samples, N_beats)
    # i_ensembles = (<N_beats,)
    # ensemble_beats = (N_samples, <N_beats)
    
    # this function looks at the past 4 beats of a beat (including itself) and average them to produce an ensemble beat
    
    ensemble_beats = []
#     i_ensembles = []
    for i_beat in range(sig_beats.shape[1]):
        if i_beat == 0:
            ensemble_beats.append(sig_beats[:, i_beat])
            continue
        if i_beat-N_enBeats < 0:
            ensemble_beats.append(sig_beats[:, :i_beat].mean(axis=1))
#             i_ensembles.append(i_R_peaks[i_beat])
        else:
            if use_woody==True:
                try:
    #                 _, _, ensAv, _, _, _ = eng.woodysMethodEvArr2(matlab.double(sig_beats[:, i_beat-N_enBeats:i_beat].tolist()), nargout=6)
                    _, _, ensAv, _, _, _ = eng.woodysMethodEvArr2(matlab.double(sig_beats[:, i_beat-N_enBeats:i_beat].tolist()), nargout=6)
                    ensAv = np.asarray(ensAv).squeeze()
                except:
                    ensAv = sig_beats[:, i_beat-N_enBeats:i_beat].mean(axis=1)
            else:
                ensAv = sig_beats[:, i_beat-N_enBeats:i_beat].mean(axis=1)
                
            ensemble_beats.append(ensAv)
#             ensemble_beats.append(sig_beats[:, i_beat-N_enBeats:i_beat].mean(axis=1))
    
    ensemble_beats = np.stack(ensemble_beats,axis=1)
    return ensemble_beats


def inspect_QRS_detector(QRS_detector_dict, subject_id, Fs, fig_name=None, outputdir=None, show_plot=False):

    ecg_dict = QRS_detector_dict['ecg_dict']
    i_R_peaks = QRS_detector_dict['i_R_peaks']
    i_S_peaks = QRS_detector_dict['i_S_peaks']
    i_beat_peaks = QRS_detector_dict['i_beat_peaks']

    fontsize = 20
    fig, ax = plt.subplots(figsize=(ecg_dict['ecg_filt1'].shape[0]/Fs/350*200, 3), dpi=80)
    
    ax.plot(np.arange(ecg_dict['ecg_filt1'].shape[0])/Fs, ecg_dict['ecg_filt1'])
    ax.plot(i_S_peaks/Fs, ecg_dict['ecg_filt1'][i_S_peaks], marker='.', color='r', alpha=0.5, label='S_peaks')
    ax.plot(i_R_peaks/Fs, ecg_dict['ecg_filt1'][i_R_peaks], marker='.', color='g', alpha=0.5, label='R_peaks')
#     ax.plot(i_beat_peaks/Fs, ecg_dict['ecg_filt1'][i_beat_peaks], marker='.', color='blue')
    
    ax.legend(frameon=True, loc='upper right', fontsize=fontsize)
    ax.set_ylabel(unit_dict['ecg'], fontsize=fontsize)        
    ax.set_xlabel('time (sec)', fontsize=fontsize)
    
    fig.tight_layout()

    if outputdir is not None:

        if fig_name is None:
            fig_name = 'inspect_sub{}'.format(subject_id)

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name+'.png', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')