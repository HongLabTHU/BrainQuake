# encoding=utf-8
import sys

from PyQt5.QtWidgets import QApplication,  QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLineEdit, QDesktopWidget, QGridLayout, QFileDialog,  QListWidget, QLabel,QFrame,QGroupBox
from PyQt5.QtCore import Qt, QThread
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.widgets import Slider

import numpy as np
from scipy.signal import spectrogram, butter, filtfilt, lfilter
from scipy.ndimage import gaussian_filter
from scipy.signal import iirnotch
from scipy.signal import convolve2d
# import h5py

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import mne
from gui_forms.ictal_form import Ictal_gui


class figure_thread(QThread):
    def __init__(self, parent=None):
        super(figure_thread, self).__init__(parent=parent)
        self.ei = parent.ei_ei

    def run(self):
        pass

class fullband_computation_thread(QThread):
    fullband_done_sig = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, raw_signal=None, ei=None, fs=2000):
        super(fullband_computation_thread, self).__init__(parent=parent)
        self.raw_signal = raw_signal
        self.fs = fs
        self.ei = ei

    def run(self):
        spec_pca, fullband_labels, fullband_ind = compute_full_band(self.raw_signal, self.fs, self.ei)
        fullband_res = [spec_pca, fullband_labels, fullband_ind]
        self.fullband_done_sig.emit(fullband_res)


def get_name_fromEdf(file_absPath):
    with open(file_absPath,'rb') as fh:
        fh.read(8)
        pinfo=fh.read(80).decode('latin-1').rstrip()
        pinfo=pinfo.split(' ')
        patient_name=pinfo[3]
    return patient_name


def compute_hfer(target_data, base_data, fs):
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy=convolve2d(target_sq,np.ones((1,window)),'same')
    base_energy=convolve2d(base_sq,np.ones((1,window)),'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    target_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, target_energy.shape[1]))
    base_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, base_energy.shape[1]))
    norm_target_energy = target_energy / target_de_matrix.astype(np.float32)
    norm_base_energy = base_energy / base_de_matrix.astype(np.float32)
    return norm_target_energy, norm_base_energy


def determine_threshold_onset(target, base):
    base_data = base.copy()
    target_data = target.copy()
    sigma = np.std(base_data, axis=1, ddof=1)
    channel_max_base = np.max(base_data, axis=1)
    thresh_value = channel_max_base + 20 * sigma
    onset_location = np.zeros(shape=(target_data.shape[0],))
    for channel_idx in range(target_data.shape[0]):
        logic_vec = target_data[channel_idx, :] > thresh_value[channel_idx]
        if np.sum(logic_vec) == 0:
            onset_location[channel_idx] = len(logic_vec)
        else:
            onset_location[channel_idx] = np.where(logic_vec != 0)[0][0]
    return onset_location


def compute_ei_index(target, base, fs):
    ei = np.zeros([1, target.shape[0]])
    hfer = np.zeros([1, target.shape[0]])
    onset_rank = np.zeros([1, target.shape[0]])
    channel_onset = determine_threshold_onset(target, base)
    seizure_location = np.min(channel_onset)
    onset_channel = np.argmin(channel_onset)
    hfer = np.sum(target[:, int(seizure_location):int(seizure_location + 0.25 * fs)], axis=1) / (fs * 0.25)
    onset_asend = np.sort(channel_onset)
    time_rank_tmp = np.argsort(channel_onset)
    onset_rank = np.argsort(time_rank_tmp) + 1
    onset_rank = np.ones((onset_rank.shape[0],)) / np.float32(onset_rank)
    ei = np.sqrt(hfer * onset_rank)
    for i in range(len(ei)):
        if np.isnan(ei[i]) or np.isinf(ei[i]):
            ei[i] = 0
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return ei, hfer, onset_rank#,channel_onset


def choose_kmeans_k(data, k_range):
    k_sse = []
    for k in k_range:
        tmp_kmeans = KMeans(n_clusters=k)
        tmp_kmeans.fit(data)
        k_sse.append(tmp_kmeans.inertia_)
    k_sse = np.array(k_sse)
    k_sseDiff = -np.diff(k_sse)
    k_sseDiffMean = np.mean(k_sseDiff)
    best_index = np.where(k_sseDiff < k_sseDiffMean)[0][0]
    return k_range[best_index]


def find_ei_cluster_ratio(pei, labels, ei_elec_num=10):
    top_elec_ind = list(np.argsort(-pei)[:ei_elec_num])
    top_elec_labels = list(labels[top_elec_ind])
    top_elec_count = {}
    top_elec_set = set(top_elec_labels)
    for i in top_elec_set:
        top_elec_count[i] = top_elec_labels.count(i)
    cluster_ind1 = [k for k, v in top_elec_count.items() if v > ei_elec_num / 2]
    if len(cluster_ind1):
        return np.array(cluster_ind1)
    else:
        cluster_ind2 = [k for k, v in top_elec_count.items() if v > ei_elec_num / 3]
        if len(cluster_ind2):
            return np.array(cluster_ind2)
        else:
            return None


def pad_zero(data, length):
    data_len = len(data)
    if data_len < length:
        # tmp_data = np.zeros(length) ### test!!!
        tmp_data = np.zeros(int(length))
        tmp_data[:data_len] = data
        return tmp_data
    return data


def cal_zscore(data):
    dmean = np.mean(data, axis=1)
    dstd = np.std(data, axis=1)
    norm_data = (data - dmean[:, None]) / dstd[:, None]
    return norm_data


def cal_specs_matrix(raw, sfreq, method='STFT'):
    win_len = 0.5
    overlap = 0.8
    freq_range = 300
    half_width = win_len * sfreq
    ch_num = raw.shape[0]
    if method == 'STFT':
        for i in range(ch_num):
            if i % 10 == 0:
                print(str(i) + '/' + str(ch_num))
            time_signal = raw[i, :].ravel()
            time_signal = pad_zero(time_signal, 2 * half_width)
            f, t, hfo_spec = spectrogram(time_signal, fs=int(sfreq), nperseg=int(half_width),
                                         noverlap=int(overlap * half_width),
                                         nfft=1024, mode='magnitude')
            hfo_new = 20 * np.log10(hfo_spec + 1e-10)
            hfo_new = gaussian_filter(hfo_new, sigma=2)
            freq_nums = int(len(f) * freq_range / f.max())
            hfo_new = hfo_new[:freq_nums, :]
            tmp_specs = np.reshape(hfo_new, (-1,))
            if i == 0:
                chan_specs = tmp_specs
            else:
                chan_specs = np.row_stack((chan_specs, tmp_specs))
    f_cut = f[:freq_range]
    return chan_specs, hfo_new.shape, t, f_cut


def norm_specs(specs):
    specs_mean = specs - specs.mean(axis=0)
    specs_norm = specs_mean / specs_mean.std(axis=0)
    return specs_norm


def compute_full_band(raw_data, sfreq, ei):
    ei_elec_num = 10
    print('computing spectrogram')
    raw_specs, spec_shape, t, f = cal_specs_matrix(raw_data, sfreq, 'STFT')
    raw_specs_norm = norm_specs(raw_specs)
    print('dimensionality reducing')
    proj_pca = PCA(n_components=10)
    # raw_specs_norm[np.where(raw_specs_norm == np.nan)] = 0
    # raw_specs_norm[np.where(raw_specs_norm == np.inf)] = 0
    spec_pca = proj_pca.fit_transform(raw_specs_norm)
    top_elec_ind = np.argsort(-ei)[:ei_elec_num]
    top_elec_pca = np.zeros([ei_elec_num, spec_pca.shape[1]])
    for i in range(ei_elec_num):
        top_elec_pca[i] = spec_pca[top_elec_ind[i]]
    print('clustering')
    k_num = choose_kmeans_k(spec_pca, range(2, 8))
    tmp_kmeans = KMeans(n_clusters=k_num)
    tmp_kmeans.fit(spec_pca)
    pre_labels = tmp_kmeans.labels_
    cluster_ind_ratio = find_ei_cluster_ratio(ei, pre_labels)

    chosen_cluster_ind = np.where(pre_labels == cluster_ind_ratio)[0]
    return spec_pca, pre_labels, chosen_cluster_ind


# main class
class IctalModule(QWidget, Ictal_gui):
    def __init__(self,parent):
        super(IctalModule, self).__init__()
        self.setupUi(self)
        self.parent=parent
        # self.initUI()

    # set functions
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    # input edf data
    def dialog_inputedfdata(self):
        self.mat_filename, b = QFileDialog.getOpenFileName(self, 'open edf file', './', '(*.edf)')
        if self.mat_filename:
            # load data
            self.patient_name = self.lineedit_patient_name.text()
            self.edf_data = mne.io.read_raw_edf(self.mat_filename, preload=True, stim_channel=None)
            self.preprocess_xw()
            self.band_low = 1.0
            self.band_high = 500
            self.edf_time_max = self.modified_edf_data.shape[1] / self.fs
            self.disp_flag = 0
            self.data_fomat = 1 #edf

            QMessageBox.information(self, '', 'data loaded')
            # init display params
            self.init_display_params()
            self.disp_refresh()

            # enable buttons
            self.reset_data_display.setEnabled(True)
            self.target_button.setEnabled(True)
            self.baseline_button.setEnabled(True)
            self.chans_del_button.setEnabled(True)
            self.filter_button.setEnabled(True)
            self.dis_up.setEnabled(True)
            self.dis_down.setEnabled(True)
            self.dis_add_mag.setEnabled(True)
            self.dis_drop_mag.setEnabled(True)
            self.dis_more_chans.setEnabled(True)
            self.dis_less_chans.setEnabled(True)
            self.dis_shrink_time.setEnabled(True)
            self.dis_expand_time.setEnabled(True)
            self.dis_left.setEnabled(True)
            self.dis_right.setEnabled(True)

    # init display
    def init_display_params(self):
        self.disp_chans_num = 20
        self.disp_chans_start = 0
        self.disp_wave_mul = 10
        self.disp_time_win = 5
        self.disp_time_start = 0

        # self.baseline_pos = np.array([0.0, int(self.edf_time_max / 5)])
        self.baseline_pos = np.array([0.0, 1.0])
        self.target_pos = np.array([0.0, self.edf_time_max])
        self.baseline_mouse = 0
        self.target_mouse = 0
        self.ei_target_start = self.target_pos[0]
        self.ei_target_end = self.target_pos[1]
        self.modified_edf_data = self.origin_data.copy()
        self.disp_ch_names = self.origin_chans.copy()
        self.chans_list.clear()
        self.chans_list.addItems(self.disp_ch_names)

        self.edf_time = self.modified_edf_data.shape[1]/self.fs
        self.edf_nchans = len(self.chans_list)
        self.edf_line_colors = np.array([cm.jet(x) for x in np.random.rand(self.edf_nchans)])
        self.edf_dmin = self.modified_edf_data[:, :].min()
        self.edf_dmax = self.modified_edf_data[:, :].max()
        self.disp_press = 0.7
        self.dr = (self.edf_dmax - self.edf_dmin) * self.disp_press
        self.y0 = self.edf_dmin
        self.y1 = (self.disp_chans_num - 1) * self.dr + self.edf_dmax
        self.disp_flag = 0

    # refresh display
    def disp_refresh(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_ylim(self.y0, self.y1)
        segs = []
        ticklocs = []
        self.disp_start = int(self.disp_time_start*self.fs)
        self.disp_end = int((self.disp_time_start + self.disp_time_win)*self.fs)
        self.disp_end=min(self.disp_end,self.modified_edf_data.shape[1])
        if self.disp_chans_num>=self.modified_edf_data.shape[0]:
            self.disp_chans_start=0
            self.disp_chans_num=self.modified_edf_data.shape[0]
        elif self.disp_chans_start+self.disp_chans_num>=self.modified_edf_data.shape[0]:
            self.disp_chans_start=self.modified_edf_data.shape[0]-self.disp_chans_num
        for i in range(self.disp_chans_start, self.disp_chans_start + self.disp_chans_num):
            tmp_data = self.modified_edf_data[i, self.disp_start:self.disp_end]

            tmp_time = np.linspace(self.disp_start/self.fs, self.disp_end/self.fs, self.disp_end-self.disp_start)
            tmp_data = tmp_data * self.disp_wave_mul
            segs.append(np.hstack((tmp_time[:, np.newaxis], tmp_data[:, np.newaxis])))
            ticklocs.append((i - self.disp_chans_start) * self.dr)
        offsets = np.zeros((self.disp_chans_num, 2), dtype=float)
        offsets[:, 1] = ticklocs
        colors = self.edf_line_colors[self.disp_chans_start:self.disp_chans_start + self.disp_chans_num]
        # linewidths=
        lines = LineCollection(segs, offsets=offsets, linewidths=0.7,transOffset=None,colors='k')  # ,colors=colors,transOffset=None)
        disp_chan_names = self.disp_ch_names[
                          self.disp_chans_start:(self.disp_chans_start + self.disp_chans_num)]
        self.canvas.axes.set_xlim(segs[0][0, 0], segs[0][-1, 0])
        self.canvas.axes.add_collection(lines)

        self.canvas.axes.set_yticks(ticklocs)
        self.canvas.axes.set_yticklabels(disp_chan_names)
        self.canvas.axes.set_ylim(self.edf_dmin, (self.disp_chans_num - 1) * self.dr + self.edf_dmax)
        self.canvas.axes.set_xlabel('time(s)')
        #add first line
        if hasattr(self,'baseline_count')  and self.baseline_count==1 and (self.baseline_pos[0]>segs[0][0,0] and self.baseline_pos[0]<segs[0][-1,0]):
            self.canvas.axes.axvline(self.baseline_pos[0])
        if hasattr(self,'target_count') and self.target_count==1 and (self.target_pos[0]>segs[0][0,0] and self.target_pos[0]<segs[0][-1,0]):
            self.canvas.axes.axvline(self.target_pos[0])
        self.canvas.draw()

    # preprecess xw
    def preprocess_xw(self):
        self.fs = self.edf_data.info['sfreq']
        self.disp_ch_names = self.edf_data.ch_names
        self.chans_list.addItems(self.disp_ch_names)
        # self.modified_edf_data, self.times = self.edf_data[:]
        self.origin_data, self.times = self.edf_data[:]
        # self.origin_data = self.modified_edf_data.copy()
        self.modified_edf_data=self.origin_data.copy()
        self.origin_chans = self.disp_ch_names.copy()

    # disp button slot functions
    def reset_data_display_func(self):
        self.target_pos = np.array([0.0, self.edf_time_max])
        self.baseline_pos = np.array([0.0, 1.0])
        self.init_display_params()
        self.disp_refresh()
        self.ei_button.setEnabled(False)
        self.hfer_button.setEnabled(False)
        self.fullband_button.setEnabled(False)

    def origin_data_display_func(self):
        self.disp_flag = 0
        self.disp_refresh()

    def disp_win_down_func(self):
        self.disp_chans_start -= self.disp_chans_num
        if self.disp_chans_start <= 0:
            self.disp_chans_start = 0
        self.disp_refresh()

    def disp_win_up_func(self):
        self.disp_chans_start += self.disp_chans_num
        # if self.disp_chans_start + self.disp_chans_num >= self.edf_nchans:
        if self.disp_chans_start + self.disp_chans_num >= self.modified_edf_data.shape[0]:
            # self.disp_chans_start = self.edf_nchans - self.disp_chans_num-1
            self.disp_chans_start = self.modified_edf_data.shape[0] - self.disp_chans_num
        self.disp_refresh()

    def disp_more_chans_func(self):
        self.disp_chans_num *= 2
        # if self.disp_chans_num >= self.edf_nchans:
        if self.disp_chans_num >= self.modified_edf_data.shape[0]:
            self.disp_chans_start=0
            self.disp_chans_num = self.modified_edf_data.shape[0]
        elif self.disp_chans_start+self.disp_chans_num>=self.modified_edf_data.shape[0]:
            self.disp_chans_start=self.modified_edf_data.shape[0]-self.disp_chans_num
        self.disp_refresh()

    def disp_less_chans_func(self):
        self.disp_chans_num = int(self.disp_chans_num / 2.0)
        if self.disp_chans_num <= 1:
            self.disp_chans_num = 1
        self.disp_refresh()

    def disp_add_mag_func(self):
        self.disp_wave_mul *= 1.5
        print(self.disp_wave_mul)
        self.disp_refresh()

    def disp_drop_mag_func(self):
        self.disp_wave_mul *= 0.75
        print(self.disp_wave_mul)
        self.disp_refresh()

    def disp_win_left_func(self):
        self.disp_time_start -= 0.2 * self.disp_time_win
        if self.disp_time_start <= 0:
            self.disp_time_start = 0
        self.disp_refresh()

    def disp_win_right_func(self):
        self.disp_time_start += 0.2 * self.disp_time_win
        if self.disp_time_start + self.disp_time_win >= self.edf_time:
            self.disp_time_start = self.edf_time - self.disp_time_win
        self.disp_refresh()

    def disp_shrink_time_func(self):
        self.disp_time_win += 2
        if self.disp_time_win >= self.edf_time:
            self.disp_time_win = self.edf_time
        self.disp_refresh()

    def disp_expand_time_func(self):
        self.disp_time_win -= 2
        if self.disp_time_win <= 2:
            self.disp_time_win = 2
        self.disp_refresh()

    def disp_scroll_mouse(self, e):
        if e.button == 'up':
            self.disp_win_left_func()
        elif e.button == 'down':
            self.disp_win_right_func()
            # ei functions

    # filter & del chans
    def filter_data(self):
        self.modified_edf_data=self.modified_edf_data-np.mean(self.modified_edf_data,axis=0)
        #notch filter
        notch_freqs=np.arange(50,151,50)
        for nf in notch_freqs:
            tb,ta=iirnotch(nf/(self.fs/2),30)
            self.modified_edf_data=filtfilt(tb,ta,self.modified_edf_data,axis=-1)
        #band filter
        self.band_low = float(self.disp_filter_low.text())
        self.band_high = float(self.disp_filter_high.text())
        nyq = self.fs/2
        b, a = butter(5, np.array([self.band_low/nyq, self.band_high/nyq]), btype = 'bandpass')
        self.modified_edf_data = filtfilt(b,a,self.modified_edf_data)
        self.disp_flag = 1
        self.disp_refresh()
        self.ei_button.setEnabled(True)
        self.hfer_button.setEnabled(True)

    def delete_chans(self):
        deleted_chans = self.chans_list.selectedItems()
        deleted_list = [i.text() for i in deleted_chans]
        deleted_ind_list = []
        for deleted_name in deleted_list:
            deleted_ind_list.append(self.disp_ch_names.index(deleted_name))
        new_modified_data = np.delete(self.modified_edf_data, deleted_ind_list, axis=0)
        self.modified_edf_data = new_modified_data
        for d_chan in deleted_list:
            self.disp_ch_names.remove(d_chan)
        self.chans_list.clear()
        self.chans_list.addItems(self.disp_ch_names)
        self.disp_refresh()

    # select base time & target time
    def choose_baseline(self):
        self.baseline_mouse = 1
        self.baseline_count = 0

    def choose_target(self):
        self.target_mouse = 1
        self.target_count = 0

    def canvas_press_button(self, e):
        if hasattr(self,'baseline_mouse') and self.baseline_mouse == 1:
            self.baseline_pos[self.baseline_count] = e.xdata
            print(e.xdata)
            self.canvas.axes.axvline(e.xdata)
            self.canvas.draw()
            self.baseline_count += 1
            if self.baseline_count == 2:
                self.baseline_mouse = 0
                print('baseline time', self.baseline_pos)
                reply = QMessageBox.question(self, 'confirm', 'confirm baseline?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    pass
                else:
                    self.baseline_pos = np.array([0.0, 1.0])
                    self.disp_refresh()
        elif hasattr(self,'target_mouse') and self.target_mouse == 1:
            self.target_pos[self.target_count] = e.xdata
            self.canvas.axes.axvline(e.xdata)
            self.canvas.draw()
            self.target_count += 1
            if self.target_count == 2:
                self.target_mouse = 0
                print('target time', self.target_pos)
                reply = QMessageBox.question(self, 'confim', 'confirm target time?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.disp_time_start = self.target_pos[0]
                    self.disp_time_win = self.target_pos[1] - self.target_pos[0]
                    self.disp_refresh()
                else:
                    self.target_pos = np.array([0.0, self.edf_time_max])
                    self.disp_refresh()
                    self.canvas.axes.axvline(self.baseline_pos[0])
                    self.canvas.axes.axvline(self.baseline_pos[1])
                    self.canvas.draw()
        else:
            pass

    # ei computation
    def ei_computation_func(self):
        # local
        QMessageBox.information(self,'','EI computation starting, please wait')
        self.ei_base_start = int(self.baseline_pos[0]*self.fs)
        self.ei_base_end = int(self.baseline_pos[1]*self.fs)
        self.ei_target_start = int(self.target_pos[0]*self.fs)
        self.ei_target_end = int(self.target_pos[1]*self.fs)

        self.ei_baseline_data = self.modified_edf_data.copy()[:, self.ei_base_start:self.ei_base_end]
        self.ei_target_data = self.modified_edf_data.copy()[:, self.ei_target_start:self.ei_target_end]
        self.ei_norm_target, self.ei_norm_base = compute_hfer(self.ei_target_data, self.ei_baseline_data, self.fs)
        self.ei_ei, self.ei_hfer, self.ei_onset_rank = compute_ei_index(self.ei_norm_target, self.ei_norm_base,
                                                                           self.fs)
        #for click-display signals
        self.tmp_origin_edf_data = self.origin_data.copy()
        remain_chInd = np.array([x in self.disp_ch_names for x in self.origin_chans])
        self.tmp_origin_remainData = self.tmp_origin_edf_data[remain_chInd]
        self.tmp_origin_remainData = self.tmp_origin_remainData - np.mean(self.tmp_origin_remainData, axis=0)
        # notch filt
        notch_freqs = np.arange(50, 151, 50)
        for nf in notch_freqs:
            tb, ta = iirnotch(nf / (self.fs / 2), 30)
            self.tmp_origin_remainData = filtfilt(tb, ta, self.tmp_origin_remainData, axis=-1)
        print('finish ei computation')
        self.fullband_button.setEnabled(True)
        self.ei_plot_xw_func()

    # hfer computation
    def hfer_computation_func(self):
        QMessageBox.information(self,'','HFER computation starting, please wait')
        self.hfer_base_start = int(self.baseline_pos[0]*self.fs)
        self.hfer_base_end = int(self.baseline_pos[1]*self.fs)
        self.hfer_target_start = int(self.target_pos[0]*self.fs)
        self.hfer_target_end = int(self.target_pos[1]*self.fs)
        self.hfer_baseline = self.modified_edf_data[:, self.hfer_base_start:self.hfer_base_end]
        self.hfer_target = self.modified_edf_data[:, self.hfer_target_start:self.hfer_target_end]
        self.norm_target, self.norm_base = compute_hfer(self.hfer_target, self.hfer_baseline, self.fs)
        hfer_fig = plt.figure('hfer')
        # hfer
        hfer_ax = hfer_fig.add_axes([0.1, 0.1, 0.7, 0.8])
        tmp_x, tmp_y = np.meshgrid(np.linspace(self.hfer_target_start, self.hfer_target_end, self.norm_target.shape[1]),
                                   np.arange(self.norm_target.shape[0] + 1))
        surf = hfer_ax.pcolormesh(tmp_x, tmp_y, self.norm_target, cmap=plt.cm.hot, vmax=50, vmin=0)
        if 'ei_channel_onset' in dir(self):
            hfer_ax.plot(self.hfer_target_start + self.ei_channel_onset, np.arange(len(self.ei_channel_onset)) + 0.5,
                         'ko')
        hfer_ax.set_xticks(np.arange(self.hfer_target_start, self.hfer_target_start + self.norm_target.shape[1], 2000))
        hfer_ax.set_xticklabels(np.rint(np.arange(self.hfer_target_start, self.hfer_target_start + self.norm_target.shape[1],
                                           2000) / float(self.fs)).astype(np.int16))
        hfer_ax.set_xlabel('time(s)')
        hfer_ax.set_ylabel('channels')
        hfer_fig.canvas.mpl_connect('button_press_event', self.hfer_press_func)
        # colorbar
        color_bar_ax = hfer_fig.add_axes([0.85, 0.1, 0.02, 0.8])
        plt.colorbar(surf, cax=color_bar_ax, orientation='vertical')
        plt.show()

    # press hfer to show original signal and spectrogram
    def hfer_press_func(self, e):
        chosen_elec_index = int(e.ydata)  # int(round(e.ydata))

        # compute spectrogram
        elec_name = self.disp_ch_names[chosen_elec_index]
        raw_data_indx = self.disp_ch_names.index(elec_name)
        tmp_origin_edf_data = self.tmp_origin_remainData
        tmp_data = tmp_origin_edf_data[raw_data_indx, self.hfer_target_start:self.hfer_target_end]
        tmp_time_target = np.linspace(self.hfer_target_start/self.fs,self.hfer_target_end/self.fs,
                                     int((self.hfer_target_end-self.hfer_target_start)))

        fig = plt.figure('signal')
        ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        ax1.cla()
        ax1.set_title(elec_name + ' signal')
        if self.data_fomat == 1:
            tmp_data_plot = tmp_data*1000
        elif self.data_fomat == 0:
            tmp_data_plot = tmp_data/1000
        ax1.plot(tmp_time_target, tmp_data_plot)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('signal(mV)')
        ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        ax1_ymax = np.abs(tmp_data_plot).max()
        ax1.set_ylim([-ax1_ymax, ax1_ymax])
        # ax2
        ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
        ax2.cla()
        ax2.set_title(elec_name + ' spectrogram')
        f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
        sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
        sxx = gaussian_filter(sxx, sigma=2)
        spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
        spec_f_max = 300
        spec_f_nums = int(len(f) * spec_f_max / f.max())
        spec_f = np.linspace(0, spec_f_max, spec_f_nums)
        spec_sxx = sxx[:spec_f_nums, :]

        spec_time, spec_f = np.meshgrid(spec_time, spec_f)
        surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8, shading='auto')

        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('frequency(hz)')
        ax2.set_ylim((0, spec_f_max))
        ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
        cb = plt.colorbar(surf, cax=position)
        plt.show()

    def ei_plot_xw_func(self):
        ei_mu = np.mean(self.ei_ei)
        ei_std = np.std(self.ei_ei)
        self.ei_thresh = ei_mu + ei_std

        self.ei_ei_fig = plt.figure('ei')
        ei_ei_ax = self.ei_ei_fig.add_subplot(111)
        ei_hfer_fig = plt.figure('hfer')
        ei_hfer_ax = ei_hfer_fig.add_subplot(111)
        ei_onset_rank_fig = plt.figure('onset')
        ei_onset_rank_ax = ei_onset_rank_fig.add_subplot(111)
        ei_data = np.stack([self.ei_hfer, self.ei_onset_rank], axis=0)
        title_data = ['High frequency Energy Coefficient', 'Time Coefficient']
        print(len(ei_data))
        ei_axes = [ei_hfer_ax, ei_onset_rank_ax]

        ei_ei_ax.bar(range(len(self.ei_ei)), self.ei_ei)
        ei_ei_ax.set_title('High Frequency Epileptogenicity Index')
        ei_ind = list(np.squeeze(np.where(self.ei_ei > self.ei_thresh)))
        print(ei_ind)
        for ind in ei_ind:
            print(ind)
            ei_ei_ax.text(ind-0.8, self.ei_ei[ind]+0.01, self.disp_ch_names[ind], fontsize=8, color='k')
        ei_ei_ax.plot(np.arange(len(self.ei_ei)), self.ei_thresh * np.ones(len(self.ei_ei)), 'r--')
        for i in range(len(ei_data)):
            ei_axes[i].bar(range(len(ei_data[i])), ei_data[i])
            ei_axes[i].set_title(title_data[i])
        self.ei_ei_fig.canvas.mpl_connect('button_press_event', self.ei_press_func)
        plt.show()


    def ei_press_func(self, e):
        if e.button == 1:
            chosen_elec_index = int(round(e.xdata))
            # compute spectrum
            elec_name = self.disp_ch_names[chosen_elec_index]
            raw_data_indx = self.disp_ch_names.index(elec_name)
            tmp_origin_edf_data = self.tmp_origin_remainData
            tmp_data = tmp_origin_edf_data[raw_data_indx, self.ei_target_start:self.ei_target_end]
            tmp_time_target = np.linspace(self.ei_target_start/self.fs, self.ei_target_end/self.fs,
                                          int((self.ei_target_end - self.ei_target_start)))

            fig = plt.figure('signal')
            ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
            ax1.cla()
            ax1.set_title(elec_name + ' signal')
            if self.data_fomat == 1:
                tmp_data_plot = tmp_data * 1000
            elif self.data_fomat == 0:
                tmp_data_plot = tmp_data/1000
            ax1.plot(tmp_time_target, tmp_data_plot)
            ax1.set_xlabel('time(s)')
            ax1.set_ylabel('signal(mV)')
            ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
            ax1_ymax = np.abs(tmp_data_plot).max()
            ax1.set_ylim([-ax1_ymax, ax1_ymax])
            # ax2
            ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
            ax2.cla()
            ax2.set_title(elec_name + ' spectrogram')
            f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                    noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
            sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
            sxx = gaussian_filter(sxx, sigma=2)
            spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
            spec_f_max = 300
            spec_f_nums = int(len(f) * spec_f_max / f.max())
            spec_f = np.linspace(0, spec_f_max, spec_f_nums)
            spec_sxx = sxx[:spec_f_nums, :]

            spec_time, spec_f = np.meshgrid(spec_time, spec_f)
            surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8, shading='auto')

            ax2.set_xlabel('time(s)')
            ax2.set_ylabel('frequency(hz)')
            ax2.set_ylim((0, spec_f_max))
            ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
            position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
            cb = plt.colorbar(surf, cax=position)
            plt.show()
        elif e.button == 3:
            self.ei_thresh = e.ydata
            print(self.ei_thresh)
            self.ei_ei_fig.clf()
            ei_ei_ax = self.ei_ei_fig.add_axes([0.1, 0.1, 0.75, 0.8])
            # ei_ei_ax = plt.axes()
            ei_ei_ax.bar(range(len(self.ei_ei)), self.ei_ei)
            ei_ei_ax.set_title('High Frequency Epileptogenicity Index')
            ei_ind = list(np.squeeze(np.where(self.ei_ei > self.ei_thresh)))
            print(ei_ind)
            for ind in ei_ind:
                print(ind)
                ei_ei_ax.text(ind - 0.8, self.ei_ei[ind] + 0.01, self.disp_ch_names[ind], fontsize=8, color='k')
            ei_ei_ax.plot(np.arange(len(self.ei_ei)), self.ei_thresh * np.ones(len(self.ei_ei)), 'r--')
            axthresh = plt.axes([0.9, 0.1, 0.02, 0.8])
            plt.show()


    # full band computation
    def fullband_computation_func(self):
        self.fullband_button.setEnabled(False)

        self.fullband_base_start = int(self.baseline_pos[0] * self.fs)
        self.fullband_base_end = int(self.baseline_pos[1] * self.fs)
        self.fullband_target_start = int(self.target_pos[0] * self.fs)
        self.fullband_target_end = int(self.target_pos[1] * self.fs)

        self.fullband_target = self.tmp_origin_remainData[:, self.fullband_target_start:self.fullband_target_end]

        QMessageBox.information(self, '', 'full band computation starting, please wait')
        self.fullband_thread = fullband_computation_thread(parent=self, raw_signal=self.fullband_target, ei=self.ei_ei,
                                                           fs=self.fs)
        self.fullband_thread.fullband_done_sig.connect(self.fullband_plot_func)
        self.fullband_thread.start()

    # full band plot function
    def fullband_plot_func(self, fullband_res):
        QMessageBox.information(self, '', 'fullband computation done')
        self.fullband_button.setEnabled(True)

        self.spec_pca = fullband_res[0]
        self.fullband_labels = fullband_res[1]
        self.fullband_ind = fullband_res[2]

        chs_labels = np.array(self.disp_ch_names)[self.fullband_ind]
        print('electrodes:', chs_labels)

        fullband_fig = plt.figure('full_band')
        fullband_ax = fullband_fig.add_subplot(111)
        fullband_fig.canvas.mpl_connect('button_press_event', self.fullband_press_func)
        fullband_ax.scatter(self.spec_pca[:, 0], self.spec_pca[:, 1], alpha=0.8, c=self.fullband_labels)

        for ind in self.fullband_ind:
            fullband_ax.text(self.spec_pca[ind, 0], self.spec_pca[ind, 1], self.disp_ch_names[ind],
                             fontsize=8, color='k')
        plt.show()

    def fullband_press_func(self, e):

        pos_x = e.xdata
        pos_y = e.ydata
        distance = np.sum((np.array(self.spec_pca[:, 0:2]) - np.array([pos_x, pos_y])) ** 2, axis=-1)
        chosen_elec_index = np.argmin(distance)

        elec_name = self.disp_ch_names[chosen_elec_index]
        raw_data_indx = self.disp_ch_names.index(elec_name)
        tmp_origin_edf_data = self.tmp_origin_remainData
        tmp_data = tmp_origin_edf_data[raw_data_indx, self.fullband_target_start:self.fullband_target_end]
        tmp_time_target = np.linspace(self.fullband_target_start / self.fs, self.fullband_target_end / self.fs,
                                      int((self.fullband_target_end - self.fullband_target_start)))

        fig = plt.figure('signal')
        ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        ax1.cla()
        ax1.set_title(elec_name + ' signal')
        if self.data_fomat == 1:
            tmp_data_plot = tmp_data*1000
        elif self.data_fomat == 0:
            tmp_data_plot = tmp_data/1000
        ax1.plot(tmp_time_target, tmp_data_plot)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('signal(mV)')
        ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        ax1_ymax = np.abs(tmp_data_plot).max()
        ax1.set_ylim([-ax1_ymax, ax1_ymax])
        # ax2
        ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
        ax2.cla()
        ax2.set_title(elec_name + ' spectrogram')
        f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
        sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
        sxx = gaussian_filter(sxx, sigma=2)
        spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
        spec_f_max = 300
        spec_f_nums = int(len(f) * spec_f_max / f.max())
        spec_f = np.linspace(0, spec_f_max, spec_f_nums)
        spec_sxx = sxx[:spec_f_nums, :]

        spec_time, spec_f = np.meshgrid(spec_time, spec_f)
        surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8, shading='auto')

        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('frequency(hz)')
        ax2.set_ylim((0, spec_f_max))
        ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
        cb = plt.colorbar(surf, cax=position)
        plt.show()



#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = Brainquake()
#     sys.exit(app.exec_())
#
