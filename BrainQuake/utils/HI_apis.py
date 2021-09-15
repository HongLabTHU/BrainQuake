import mne
import numpy as np
import os
import gc
# from HI_utils import *
from .interictal_utils import *

segment_time=50

def HI_preprocess_file(filename,remain_chns,highpass_freqband,proBar):
    filedir=os.path.dirname(filename)
    fileBaseName=os.path.basename(filename)
    filePreExt=fileBaseName.split('.')[0]
    fileResultsDir='./HFOdets/'+filePreExt
    if os.path.exists(fileResultsDir):
        os.system('rm -r '+fileResultsDir)
        os.makedirs(fileResultsDir)
    else:
        os.makedirs(fileResultsDir)

    edf_data = mne.io.read_raw_edf(filename, preload=False, stim_channel=None)
    fs = edf_data.info['sfreq']

    valid_chns_index=np.arange(len(edf_data.ch_names))[np.array([x in remain_chns for x in edf_data.ch_names])]
    valid_chns=np.array(edf_data.ch_names)[valid_chns_index]
    valid_chns_st=valid_chns


    time_inter=np.arange(0,edf_data.times[-1],segment_time)
    time_inter=np.arange(0,edf_data.times[-1],segment_time)
    time_inter=np.append(time_inter,edf_data.times[-1])
    time_ranges=np.array(list(zip(time_inter[:-1],time_inter[1:])))

    for id,tr in enumerate(time_ranges):
        print('part {}/{}'.format(id+1,time_ranges.shape[0]))
        start,end=edf_data.time_as_index(tr)
        batch_data=edf_data[valid_chns_index,start:end][0]
        batch_data=batch_data-batch_data.mean(axis=0)
        batch_data=notch_filt(batch_data,fs,np.arange(50,highpass_freqband[1]+10,50))
        batch_enve = return_hil_enve_norm(batch_data, fs, highpass_freqband)
        batch_t=np.arange(batch_enve.shape[1])/fs+tr[0]

        np.savez(os.path.join(fileResultsDir,'rawEnve_{}.npz'.format(id+1)),rawEnve=batch_enve,rawTimes=batch_t,valid_chns_index=valid_chns_index,
                 valid_chns=valid_chns_st,fs=fs)

        del batch_data,batch_enve
        gc.collect()
        proBar.setValue(90*(id+1)/time_ranges.shape[0])



def HI_count_highEvents_chns(filename,rel_thresh,abs_thresh,min_gap,min_last):
    filedir=os.path.dirname(filename)
    fileBaseName=os.path.basename(filename)
    filePreExt=fileBaseName.split('.')[0]
    fileResultsDir='./HFOdets/'+filePreExt


    file_highEnve_times,file_highEnve_chnsCount,file_chnsNames=find_high_enveTimes_dir(fileResultsDir,segment_time,rel_thresh=rel_thresh,
                                        abs_thresh=abs_thresh,min_gap=min_gap,min_last=min_last)

    np.savez(os.path.join('./HFOdets/',filePreExt+'_events.npz'),file_highEventsCount=file_highEnve_chnsCount,file_chnsNames=file_chnsNames,
             file_highEvents_times=file_highEnve_times)
    os.system('rm -r '+fileResultsDir)
    os.system('trash-empty')

    return [file_highEnve_chnsCount,file_chnsNames,file_highEnve_times]





