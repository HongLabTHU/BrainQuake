import numpy as np
from scipy.signal import butter,filtfilt

import os
from functools import reduce
from scipy import signal
from scipy import fftpack

def notch_filt(data,fs,freqs):
    nyq=fs/2
    Q=30
    tmp_data=data.copy()
    for f in freqs:
        tmp_w=f/(nyq)
        b,a=signal.iirnotch(tmp_w,Q)
        tmp_data=filtfilt(b,a,tmp_data,axis=-1)

    return tmp_data


def band_filt(data,fs,freqband):
    nyq=fs/2
    b,a=butter(3,[freqband[0]/nyq,freqband[1]/nyq],btype='bandpass')
    return filtfilt(b,a,data,axis=-1)


hilbert3=lambda x:signal.hilbert(x,N=fftpack.next_fast_len(x.shape[-1]),axis=-1)[...,:x.shape[-1]]
def return_hil_enve(data,fs,freqband):
    filt_data=band_filt(data,fs,freqband)
    return np.abs(hilbert3(filt_data))

def return_hil_enve_norm(data,fs,freqband):
    if freqband[1]-freqband[0]<=20:
        return return_hil_enve(data,fs,freqband)
    else:
        filter_bank=np.arange(freqband[0],freqband[1],20)
        filter_bank=np.append(filter_bank,freqband[1])
        filter_bank=list(zip(filter_bank[:-1],filter_bank[1:]))
        multi_band_enve=[]
        for freq in filter_bank:
            tmp_enve=return_hil_enve(data,fs,freq)
            multi_band_enve.append(tmp_enve)
        return np.sum(multi_band_enve,axis=0)


def return_timeRanges(onOff_array,fs,start_time=0):
    times=np.arange(len(onOff_array))/fs+start_time
    start_index=np.where(np.diff(onOff_array)==1)[0]+1
    end_index=np.where(np.diff(onOff_array)==-1)[0]
    if onOff_array[0]==1:
        start_index=np.append(start_index[::-1],[0])[::-1]
    if onOff_array[-1]==1:
        end_index=np.append(end_index,[len(onOff_array)-1])

    if len(start_index)==0 or len(end_index)==0:
        return np.array([])
    range_times=np.vstack([times[start_index],times[end_index]]).T
    return range_times

def merge_timeRanges(range_times,min_gap=10):
    merged_times=[]
    range_times=range_times.tolist()
    if len(range_times)==0:
        return []
    merged_times.append(range_times[0])
    for i in range(1,len(range_times)):
        if range_times[i][0]-merged_times[-1][1]<min_gap*1e-3:
            merged_times[-1][1]=range_times[i][1]
        else:
            merged_times.append(range_times[i])
    return merged_times

def find_high_enveTimes(raw_enve,chns_names,fs,rel_thresh=3.,abs_thresh=3.,min_gap=20,min_last=50,start_time=0):
    whole_data_median=np.median(raw_enve)
    high_times=[]
    for chi in range(len(chns_names)):
        tmp_enve=raw_enve[chi]
        tmp_std=np.std(tmp_enve)
        tmp_median=np.median(tmp_enve)
        tmp_highTime=((tmp_enve>rel_thresh*tmp_median)&(tmp_enve>abs_thresh*whole_data_median)).astype('int')
        tmp_highTime=return_timeRanges(tmp_highTime,fs,start_time)
        tmp_highTime=merge_timeRanges(tmp_highTime,min_gap)
        tmp_highEnveLong=[x[1]-x[0] for x in tmp_highTime]
        further_index=np.where((np.array(tmp_highEnveLong)>min_last*1e-3))[0]
        if len(further_index)==0:
            high_times.append([])
        else:
            tmp_highTime=np.array(tmp_highTime)[further_index]
            high_times.append(tmp_highTime.tolist())

    return high_times

def cat_chns_times(times_1,times_2):
    cat_times=[]
    for chi in range(len(times_1)):
        cat_times.append(times_1[chi]+times_2[chi])
    return cat_times

def find_high_enveTimes_dir(enve_dir,segment_time=200,rel_thresh=3.0,abs_thresh=3.,min_gap=20,min_last=50):
    whole_enveTimes=[]
    for filename in os.listdir(enve_dir):
        if filename.split('_')[0]=='rawEnve':
            tmp_filename=os.path.join(enve_dir,filename)
            tmp_enveResults=np.load(tmp_filename)
            seg_enve=tmp_enveResults['rawEnve']
            seg_chNames=tmp_enveResults['valid_chns']
            seg_fs=tmp_enveResults['fs']
            seg_startTime=(int(filename.split('.')[0].split('_')[1])-1)*segment_time
            seg_highTimes=find_high_enveTimes(seg_enve,seg_chNames,seg_fs,rel_thresh=rel_thresh,abs_thresh=abs_thresh,
                                              min_gap=min_gap,min_last=min_last,start_time=seg_startTime)
            whole_enveTimes.append(seg_highTimes)

    whole_enveTimes_cat=reduce(cat_chns_times,whole_enveTimes)
    whole_enveTimes_cat=[sorted(x,key=lambda x:x[0]) for x in whole_enveTimes_cat]

    chns_highEnve_cout=np.array([len(x) for x in whole_enveTimes_cat])

    return whole_enveTimes_cat,chns_highEnve_cout,seg_chNames


