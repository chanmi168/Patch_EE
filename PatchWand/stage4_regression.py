import numpy as np
import os
import math
from math import sin

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.ticker as plticker

import torch

import wandb

from filters import *
from setting import *
from preprocessing import *
from plotting_tools import *
from evaluate import *

# from models import *
# from models_resnet import *
# from dataset_util import *

from plotting_tools import *


def plot_feature_importances(feature_names, feature_importances, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

    # fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=100)
    fig, ax = plt.subplots(1,1, figsize=(5,feature_names.shape[0]/4), dpi=100)
    fontsize = 12
    ax.barh(feature_names, feature_importances)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax_no_top_right(ax)

    fig.tight_layout()
    
    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'feature_importance'
        else:
            fig_name = fig_name

        fig.savefig(outputdir + fig_name, bbox_inches='tight', transparent=False)

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})
        
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

def ax_conditioning_regression(ax, df_outputlabel, training_params, task_name=None, est_name='label_est', label_name='label', subject_id='All', fontsize=15):

    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)

    if task_name is None:
        task = df_outputlabel['task'].unique()[0]
        task_name = task.split('-')[0].split('_')[0]

    label = df_outputlabel[label_name].values
    label_est = df_outputlabel[est_name].values
    
    if 'label_range' in training_params:
        if training_params['label_range'] == 'label':
#             print(training_params['label_range'])
            label_range_sub = [my_floor(label.min()), my_ceil(label.max())]
        elif training_params['label_range'] == 'label+estimated':
#             print(training_params['label_range'])
            label_range_sub = [my_floor( min([label.min(), label_est.min()]) )-1, my_ceil( max([label.max(), label_est.max()]) )+1]
        elif ',' in training_params['label_range']:
            label_range_sub = np.asarray(list(training_params['label_range'].split(','))).astype(int)
    else:
#         print(training_params['label_range'])
        # label_range_sub = [my_floor(label.min()), my_ceil(label.max())]
        # label_range_sub = [my_floor( min([label.min(), label_est.min()]) ), my_ceil( max([label.max(), label_est.max()]) )]
        label_range_sub = [ min([label.min(), label_est.min()]) , max([label.max(), label_est.max()]) ]
        label_range_sub = [ label_range_sub[0] - (label_range_sub[1]-label_range_sub[0])*0.05, label_range_sub[1] + (label_range_sub[1]-label_range_sub[0])*0.05 ]

#     print(label_range_sub, label.min(), my_floor([label.min(), label_est.min()]))
#     sys.exit()
    # print(training_params)
    # print(label.min(), label_est.min(), label.max(), label_est.max())
    # print(label_range_sub)
    # sys.exit()
    
    label_range = label_range_sub

    
#     label_range_sub = [my_floor(label.min()), my_ceil(label.max())]
#     if label_range is None:
#         label_range = label_range_sub
        

    N_sub = len(df_outputlabel['CV'].unique())
    N_samples = df_outputlabel.shape[0]
    t_dur = N_samples*3/60

    PCC = get_PCC(label, label_est)
    Rsquared = get_CoeffDeterm(label, label_est)
    MAE, MAE_std = get_MAE(label, label_est)
    RMSE = get_RMSE(label, label_est)
    MAPE, MAPE_std = get_MAPE(label, label_est)
    
    title_str = '[{}]\n{} range: {:.1f}-{:.1f} {}'.format(subject_id, task_name, label_range_sub[0], label_range_sub[1], unit_dict[task_name])
    textstr = 'RMSE={:.2f} {}\nMAE={:.2f} {}\nMAPE={:.2f} {}\nPCC={:.2f}\nR2={:.2f}\nN_sub={}\nN_samples={}\nduration={:.2f} min'.format(
        RMSE, unit_dict[task_name], MAE, unit_dict[task_name],MAPE*100, '%',
        PCC, Rsquared,
        N_sub, N_samples, t_dur)
    
    ax.set_title(title_str, fontsize=fontsize+1)

#     ax.set_ylabel('Est. {} - Ref. {}\n[{}]'.format(task_name, task_name, unit_dict[task_name]), fontsize=fontsize)
#     ax.set_xlabel('Avg of Est. and Ref. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)

    ax.set_ylabel('Est. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)
    ax.set_xlabel('Ref. {}\n[{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)

    

#     major_ticks = np.arange(0, 60, 10)    
#     minor_ticks = np.arange(0, 60, 1)
#     ax.set_xticks(major_ticks)
#     ax.set_xticks(minor_ticks, minor=True)
#     ax.set_yticks(major_ticks)
#     ax.set_yticks(minor_ticks, minor=True)
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3, axis='both')
    ax.grid(which='major', alpha=0.8, axis='both')

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.plot( label_range,label_range , color='gray', alpha=0.5, linestyle='--')

    # place a text box in bottom right in axes coords
    txt_frame = ax.text(0.05, 0.65, textstr, transform=ax.transAxes, fontsize=fontsize-7,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)

    

    if 'show_metrics' in training_params:
        if training_params['show_metrics']==False:
            txt_frame.remove()
                
    ax.set_ylim(label_range)
    ax.set_xlim(label_range)

    ax_no_top_right(ax)

    
# def plot_regression(df_outputlabel_val, df_performance_val, task, training_params, est_name='label_est', label_name='label', fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
def plot_regression(df_outputlabel_val, training_params, task_name=None, est_name='label_est', label_name='label', single_color=False, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
#     print('regression')
    fig, ax = plt.subplots(1,1, figsize=(5, 5), dpi=220, facecolor='white')
    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)
    fontsize = 16
    alpha=0.4

    if task_name is None:
        task = df_outputlabel_val['task'].unique()[0]
        task_name = task.split('-')[0].split('_')[0]

    if single_color:
        sc = sns.scatterplot(data=df_outputlabel_val, x=label_name, y=est_name,  ec="None", color='#13294B', alpha=0.1, s=50, marker='o', ax=ax)
    else:
        sc = sns.scatterplot(data=df_outputlabel_val, x=label_name, y=est_name, hue='CV',  ec="None", palette=subject_palette, alpha=alpha, s=50, marker='o', ax=ax)
        ax.legend(frameon=True, fontsize=fontsize-7, bbox_to_anchor=(1.01, 1))
    # sc.set_edgecolor("none")

    
    
    ax_conditioning_regression(ax, df_outputlabel_val, training_params, task_name=task_name, est_name=est_name, label_name=label_name)


#     if 'show_metrics' in training_params:
#         if training_params['show_metrics']==False:
#             for txt in fig.texts:
#                 txt.set_visible(False)

#     # # place a text box in bottom right in axes coords
#     ax.text(0.05, 0.65, textstr, transform=ax.transAxes, fontsize=fontsize-7,
#             verticalalignment='bottom', horizontalalignment='left', bbox=props)

#     ax.set_ylim(label_range)
#     ax.set_xlim(label_range)

#     ax_no_top_right(ax)

    fig.tight_layout()
    
    
    if fig_name is None:
        fig_name = 'regression_analysis'
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
#     return fig

# def BA_plotter(ax, df, mode):
def plot_BA(df_outputlabel_val, training_params=None, task_name=None, single_color=False, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
    fig, ax = plt.subplots(1,1, figsize=(6.5, 5), dpi=220, facecolor='white')

    if task_name is None:
        task = df_outputlabel_val['task'].unique()[0]
        task_name = task.split('-')[0].split('_')[0]
    # task_name = task.split('_')[0]

    label = df_outputlabel_val['label']
    label_est =  df_outputlabel_val['label_est']
    
    if training_params is not None:
        if 'label_range' in training_params:
            if training_params['label_range'] == 'label':
    #             print(training_params['label_range'])
                label_range = [my_floor(label.min()), my_ceil(label.max())]
            elif training_params['label_range'] == 'label+estimated':
    #             print(training_params['label_range'])
                label_range = [my_floor( min([label.min(), label_est.min()]) )-1, my_ceil( max([label.max(), label_est.max()]) )+1]
            elif ',' in training_params['label_range']:
                label_range = np.asarray(list(training_params['label_range'].split(','))).astype(int)
        else:
            label_range = [my_floor( min([label.min(), label_est.min()]) )-1, my_ceil( max([label.max(), label_est.max()]) )+1]
    else:
        label_range = [my_floor( min([label.min(), label_est.min()]) )-1, my_ceil( max([label.max(), label_est.max()]) )+1]


    fontsize = 16
    alpha=0.7
        # plot Bland-Altman plot

    data1     = label
    data2     = label_est
    mean      = np.mean([data1, data2], axis=0)
    diff      = data2 - data1                    # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    diff_max = np.abs(diff).max()
    #     # plot running average of the error along x-axis (TBD)
    #     plot_smooth_err(ax, mean, diff)


    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')


    # task_name = task.split('-')[0].split('_')[0]

    if single_color:
        label = df_outputlabel_val['label']
        label_est =  df_outputlabel_val['label_est']

        mean = np.mean([label_est, label], axis=0)
        diff = label_est - label

        ax.scatter(x=mean,y=diff, alpha=0.1, color='#13294B')

    else:
    
    
        for subject_id in df_outputlabel_val['CV'].unique():

            df_sub = df_outputlabel_val[df_outputlabel_val['CV']==subject_id]
            label = df_sub['label']
            label_est =  df_sub['label_est']

            mean = np.mean([label_est, label], axis=0)
            diff = label_est - label

            ax.scatter(x=mean,y=diff, alpha=alpha, color=subject_palette[subject_id])


        
    # Annotate mean line with mean difference.
    ax.annotate('mean diff:\n{}'.format(np.round(md, 2)),
                xy=(1.2, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

    sd_limit=1.96
    if sd_limit > 0:
        half_ylim = (2 * sd_limit) * sd
#         ax.set_ylim(mean_diff - half_ylim,
#                     mean_diff + half_ylim)

        limit_of_agreement = sd_limit * sd
        lower = md - limit_of_agreement
        upper = md + limit_of_agreement
#         for j, lim in enumerate([lower, upper]):
#             ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}:\n{}'.format(sd_limit, np.round(lower, 2)),
                    xy=(1.2, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
        ax.annotate('+SD{}:\n{}'.format(sd_limit, np.round(upper, 2)),
                    xy=(1.2, 0.85),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')
        
#         print(lower, upper)
        
        ax.set_ylim(md - half_ylim,
                    md + half_ylim)

        
    else:
        ax.set_ylim(-diff_max-1,diff_max+1)


    ax.set_ylabel('Estimated {} - Reference {}\n[{}]'.format(task_name, task_name, unit_dict[task_name]), fontsize=fontsize)
    ax.set_xlabel('Average of Estimated and Reference {} [{}]'.format(task_name, unit_dict[task_name]), fontsize=fontsize)
    ax.set_title('Bland-Altman', fontsize=fontsize)

    ax.grid(which='major', axis='x', alpha=0.8)

    #     ax.xaxis.grid(True, which='major')
    #     ax.xaxis.grid(True, which='minor', alpha=0.4)
    #     ax.yaxis.grid(True, which='major')
    #     ax.yaxis.grid(True, which='minor', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlim(label_range)

    #     ax.figure.set_size_inches(14, 7)
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax_no_top_right(ax)


    fig.tight_layout()
    
    if fig_name is None:
        fig_name = 'BA_analysis'
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

#     return fig


def get_df_performance(label, label_est, subject_id, task):

    rmse = np.sqrt(mean_squared_error(label, label_est))

    mae, _ = get_MAE(label, label_est)
    mape, _ = get_MAPE(label, label_est)

    Rsquared = get_CoeffDeterm(label=label, predictions=label_est)
    PCC = get_PCC(label=label, est=label_est)

    df_performance = pd.DataFrame({
        'CV': [subject_id],
        'task': [task],
        'Rsquared': [Rsquared],
        'PCC': [PCC],
        'rmse': [rmse],
        'mae': [mae],
        'mape': [mape],
    })

    return df_performance
        
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

def plot_output(df_outputlabel, task_name=None, fig_name=None, show_plot=False, outputdir=None):
#     print('output')

    if task_name is None:
        task = df_outputlabel['task'].unique()[0]
        task_name = task.split('-')[0].split('_')[0]
        
    # fig, (ax, ax2) = plt.subplots(2,1, figsize=(20,4), dpi=80)
    fig, ax = plt.subplots(1,1, figsize=(df_outputlabel.shape[0]/25,3), dpi=100)

    x_sample = np.arange(df_outputlabel.shape[0])
    ax.plot(x_sample, df_outputlabel['label_est'], label='estimated', alpha=0.6)
    ax.plot(x_sample, df_outputlabel['label'], label='label', alpha=0.6)
    # ax2.plot(x_sample, df_outputlabel_val['CV'])

#     task_name = task.split('_')[0]
    # task_name = task.split('-')[0].split('_')[0]

    ax.set_xlim(x_sample.min(), x_sample.max()) # remove the weird white space at the beg and end of the plot

    ax.set_xlabel('sample')
    ax.set_ylabel('{}\n[{}]'.format(task_name, unit_dict[task_name]))
    ax.legend(loc='upper right')
    
    fig.tight_layout()

    if fig_name is None:
        fig_name = 'outputINtime'

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

        
        
        
def plot_regression_partial(ax, df_outputlabel, subject_id_plt, training_params, task_name=None, outputdir=None, show_plot=False, log_wandb=False):

    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.7)
    fontsize = 16

    N_beats_val = 0
    
    label = df_outputlabel['label'].values
#     label_range = [my_floor(label.min()), my_ceil(label.max())]

    for subject_id in df_outputlabel['CV'].unique():
        
        df_outputlabel_sub = df_outputlabel[df_outputlabel['CV']==subject_id]
        marker = marker_dict['circle']

        label = df_outputlabel_sub['label'].values
        label_est = df_outputlabel_sub['label_est'].values


        if subject_id == subject_id_plt:
            color = subject_palette[subject_id]
            alpha=0.4
            ax.set_title('{}'.format(subject_id), fontsize=fontsize+5)
        else:
            color = 'gray'
            alpha=0.05

        ax.scatter(label, label_est, alpha=alpha, color=color, marker=marker)
        
        
    df_outputlabel_sub = df_outputlabel[df_outputlabel['CV']==subject_id_plt]
    ax_conditioning_regression(ax, df_outputlabel_sub, training_params, task_name=task_name, subject_id=subject_id_plt)
    

# def plot_regression_all_agg(df_outputlabel, df_performance, training_params, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):
def plot_regression_all_agg(df_outputlabel, training_params, task_name=None, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

    if task_name is None:
        task = df_outputlabel['task'].unique()[0]
        task_name = task.split('-')[0].split('_')[0]
        
        
    label = df_outputlabel['label'].values
#     label_est = df_outputlabel['label_est'].values
#     label_range = [my_floor(label.min()), my_ceil(label.max())]

    
    fig = plt.figure(figsize=(20, 20), dpi=100, facecolor='white')

    for k, subject_id_plt in enumerate(df_outputlabel['CV'].unique()):
#         row = k//5
#         col = k%5

        ax = fig.add_subplot(5,5,k+1)
        plot_regression_partial(ax, df_outputlabel, subject_id_plt, training_params, outputdir=None, show_plot=False, log_wandb=False)
#         ax.set_ylim(label_range)
#         ax.set_xlim(label_range)
        
    # plot all regression in one plot
    ax = fig.add_subplot(5,5,k+2)
    alpha=0.3
    fontsize=16
    sc = sns.scatterplot(data=df_outputlabel, x='label', y='label_est', hue='CV',  ec="None", palette=subject_palette, alpha=alpha, s=50, marker='o', ax=ax)
    ax_conditioning_regression(ax, df_outputlabel, training_params)
#     ax.set_ylim(label_range)
#     ax.set_xlim(label_range)
    
    ax.legend(frameon=True, fontsize=fontsize-7, bbox_to_anchor=(1.01, 1))

    fig.tight_layout()


    if fig_name is None:
        fig_name = 'LinearR_agg'
#     else:
#         fig_name = 'LinearR_agg'+fig_name

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name+'.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
