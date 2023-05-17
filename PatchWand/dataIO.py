import os
import sys
import numpy as np

def data_saver(data, name, outputdir):
    """ usage: data_saver(df_merged_interp_alldicts, 'data', outputdir)"""
    outputdir_data = os.path.join(outputdir, name+'.npz')
    #   print('outputdir for {}:'.format(name), outputdir_data)
    np.savez(outputdir_data, data=data, allow_pickle=True)
    loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
    #     loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
    #   print('Are {} save and loadded correctly? '.format(name), np.array_equal(loaded_data, data))
    #   print('')
    
def data_loader(name, inputdir):
    """ usage: data = data_loader('data', outputdir)"""
    inputdir_data = os.path.join(inputdir, name+'.npz')
    data = np.load(inputdir_data, allow_pickle=True)['data']
    return data