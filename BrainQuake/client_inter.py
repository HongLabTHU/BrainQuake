#! /usr/bin/python3.7
# encoding=utf-8
import sys

from PyQt5.QtWidgets import QApplication,  QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLineEdit, QDesktopWidget, QGridLayout, QFileDialog,  QListWidget, QLabel,QFrame,QGroupBox,QProgressBar
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

import mne
import os

# from HI_apis import HI_preprocess_file,HI_count_highEvents_chns
from utils.HI_apis import HI_preprocess_file,HI_count_highEvents_chns
from gui_forms.inter_form import Interictal_gui


class figure_thread(QThread):
    def __init__(self, parent=None):
        super(figure_thread, self).__init__(parent=parent)
        self.ei = parent.ei_ei

    def run(self):
        pass

class HI_computation_thread(QThread):
    HI_done_sig=QtCore.pyqtSignal(object)

    def __init__(self,parent=None,interIctalFile=None,relThr=2.0,absThr=2.0,minGap=20,minDur=50,freqband=[80,250],chns_list=None,proBar=None):
        super(HI_computation_thread,self).__init__(parent=parent)
        self.interIctalFile=interIctalFile
        self.relThr=relThr
        self.absThr=absThr
        self.minGap=minGap
        self.minDur=minDur
        self.chns_list=chns_list
        self.freqband=freqband
        self.proBar=proBar

    def run(self):
        HI_preprocess_file(self.interIctalFile,self.chns_list,self.freqband,self.proBar)
        HI_results=HI_count_highEvents_chns(self.interIctalFile,self.relThr,self.absThr,self.minGap,self.minDur)
        self.proBar.setValue(100)
        self.HI_done_sig.emit(HI_results)


# main class
class InterModule(QWidget, Interictal_gui):
    def __init__(self,parent):
    # def __init__(self):
        super(InterModule, self).__init__()
        self.setupUi(self)
        self.parent=parent

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
            self.band_high = self.fs/2-1
            self.edf_time_max = self.modified_edf_data.shape[1] / self.fs

            QMessageBox.information(self, '', 'data loaded')
            # init display params
            self.init_display_params()
            self.disp_refresh()

            # enable buttons
            self.reset_data_display.setEnabled(True)
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
            self.HI_button.setEnabled(True)


    def preprocess_xw(self):
        self.fs = self.edf_data.info['sfreq']
        self.disp_ch_names = self.edf_data.ch_names
        self.chans_list.addItems(self.disp_ch_names)
        self.origin_data, self.times = self.edf_data[:]
        self.modified_edf_data = self.origin_data.copy()
        self.origin_chans = self.disp_ch_names.copy()

        # disp button slot functions

    # init display
    def init_display_params(self):
        self.disp_chans_num = 20
        self.disp_chans_start = 0
        self.disp_wave_mul = 10
        self.disp_time_win = 5
        self.disp_time_start = 0

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


    # refresh display
    def disp_refresh_ori(self):
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
        self.canvas.axes.set_ylim(-self.dr/2, (self.disp_chans_num - 1) * self.dr + self.dr/2)
        self.canvas.axes.set_xlabel('time(s)')
        #add first line
        self.canvas.draw()


    def reset_data_display_func(self):
        self.init_display_params()
        self.disp_refresh=self.disp_refresh_ori
        self.disp_refresh()

    def disp_win_down_func(self):
        self.disp_chans_start -= self.disp_chans_num
        if self.disp_chans_start <= 0:
            self.disp_chans_start = 0
        self.disp_refresh()

    def disp_win_up_func(self):
        self.disp_chans_start += self.disp_chans_num
        if self.disp_chans_start + self.disp_chans_num >= self.modified_edf_data.shape[0]:
            self.disp_chans_start = self.modified_edf_data.shape[0] - self.disp_chans_num
        self.disp_refresh()

    def disp_more_chans_func(self):
        self.disp_chans_num *= 2
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
        self.band_low = float(self.disp_filter_low.text())
        self.band_high = float(self.disp_filter_high.text())
        notch_freqs=np.arange(50,self.band_high+10,50)
        for nf in notch_freqs:
            tb,ta=iirnotch(nf/(self.fs/2),30)
            self.modified_edf_data=filtfilt(tb,ta,self.modified_edf_data,axis=-1)
        #band filter
        nyq = self.fs/2
        b, a = butter(5, np.array([self.band_low/nyq, self.band_high/nyq]), btype = 'bandpass')
        self.modified_edf_data = filtfilt(b,a,self.modified_edf_data)
        self.disp_wave_mul=self.dr/(self.modified_edf_data.std()*10)
        self.disp_refresh()



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

    def get_HI_compu_params(self):
        self.relThr_val=float(self.lineedit_relTh_name.text())
        self.absThr_val=float(self.lineedit_absTh_name.text())
        self.minGap_val=float(self.lineedit_minGap_name.text())
        self.minDur_val=float(self.lineedit_minDur_name.text())
        # self.band_low,self.band_high
        self.chnList_val=[]
        remain_chns_count=self.chans_list.count()
        for ci in range(remain_chns_count):
            self.chnList_val.append(self.chans_list.item(ci).text())

    def HI_computation_func(self):
        self.interictal_file=self.mat_filename
        if self.interictal_file:
            self.HI_button.setEnabled(False)
            self.get_HI_compu_params()
            QMessageBox.information(self,'','High frequency events Index computation starting, please wait')
            self.HI_proBar.setValue(0)
            self.HI_thread=HI_computation_thread(parent=self,interIctalFile=self.interictal_file,relThr=self.relThr_val,absThr=self.absThr_val,minGap=self.minGap_val,
                                                 minDur=self.minDur_val,freqband=[self.band_low,self.band_high],chns_list=self.chnList_val,proBar=self.HI_proBar)
            self.HI_thread.HI_done_sig.connect(self.HI_plot_func)
            self.HI_thread.start()

        else:
            QMessageBox.information(self,'','please select an interictal edf file')


    def HI_plot_func(self,HI_results):
        QMessageBox.information(self,'','HI computation done')
        self.HI_button.setEnabled(True)
        self.hiDetsFilt_button.clicked.connect(self.reset_refresh_filt)
        self.hiDetsFilt_button.setEnabled(True)
        self.hiDetsRaw_button.clicked.connect(self.reset_refresh_raw)
        self.hiDetsRaw_button.setEnabled(True)

        self.HI_chnCounts=HI_results[0]
        self.HI_chnNames=HI_results[1]
        HI_fig=plt.figure('interIctal HI')
        HI_ax=HI_fig.add_subplot(111)
        HI_ax.bar(np.arange(len(self.HI_chnCounts)),self.HI_chnCounts,color=(50/255, 168/255, 82/255))
        for chi,chName in enumerate(self.HI_chnNames):
            HI_ax.text(chi,self.HI_chnCounts[chi],chName,va='bottom',ha='center')
        plt.xlabel('Channels')
        plt.ylabel('HI')

        plt.show()

    def reset_refresh_filt(self):
        self.disp_refresh=self.disp_refresh_HFOdets_filt
        self.disp_wave_mul=self.dr/(self.modified_edf_data.std()*10)
        self.disp_refresh()

    def reset_refresh_raw(self):
        self.disp_refresh=self.disp_refresh_HFOdets_raw
        self.disp_wave_mul=self.dr/(np.median(np.std(self.origin_data,axis=1))*2)
        self.disp_refresh()

    def disp_refresh_HFOdets_filt(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_ylim(self.y0, self.y1)
        segs = []
        ticklocs = []
        self.disp_start = int(self.disp_time_start * self.fs)
        self.disp_end = int((self.disp_time_start + self.disp_time_win) * self.fs)
        self.disp_end = min(self.disp_end, self.modified_edf_data.shape[1])
        if self.disp_chans_num >= self.modified_edf_data.shape[0]:
            self.disp_chans_start = 0
            self.disp_chans_num = self.modified_edf_data.shape[0]
        elif self.disp_chans_start + self.disp_chans_num >= self.modified_edf_data.shape[0]:
            self.disp_chans_start = self.modified_edf_data.shape[0] - self.disp_chans_num
        for i in range(self.disp_chans_start, self.disp_chans_start + self.disp_chans_num):
            tmp_data = self.modified_edf_data[i, self.disp_start:self.disp_end]

            tmp_time = np.linspace(self.disp_start / self.fs, self.disp_end / self.fs, self.disp_end - self.disp_start)
            tmp_data = tmp_data * self.disp_wave_mul
            segs.append(np.hstack((tmp_time[:, np.newaxis], tmp_data[:, np.newaxis])))
            ticklocs.append((i - self.disp_chans_start) * self.dr)
        offsets = np.zeros((self.disp_chans_num, 2), dtype=float)
        offsets[:, 1] = ticklocs
        colors = self.edf_line_colors[self.disp_chans_start:self.disp_chans_start + self.disp_chans_num]
        # linewidths=
        lines = LineCollection(segs, offsets=offsets, linewidths=0.7,
                               transOffset=None,colors='k')  # ,colors=colors,transOffset=None)
        disp_chan_names = self.disp_ch_names[
                          self.disp_chans_start:(self.disp_chans_start + self.disp_chans_num)]
        self.canvas.axes.set_xlim(segs[0][0, 0], segs[0][-1, 0])
        self.canvas.axes.add_collection(lines)
        #add hfo detections
        self.hfoDets_resultsFile=os.path.join('./HFOdets',os.path.basename(self.mat_filename).split('.')[0]+'_events.npz')
        if os.path.exists(self.hfoDets_resultsFile):
            self.hfoDets=np.load(self.hfoDets_resultsFile,allow_pickle=True)
            self.hfoDets_chns=self.hfoDets['file_chnsNames'].tolist()
            self.hfoDets_times=self.hfoDets['file_highEvents_times']

        showDets_index=[self.hfoDets_chns.index(x) if  x in self.hfoDets_chns else [] for x in disp_chan_names]
        showDets_times=[self.hfoDets_times[x] for x in showDets_index]
        for ci in range(len(showDets_times)):
            if len(showDets_times[ci])==0:
                continue
            for ti,tw in enumerate(showDets_times[ci]):
                if tw[0]>(self.disp_start/self.fs) and tw[1]<(self.disp_end/self.fs):
                    self.canvas.axes.plot([tw[0],tw[1]],[self.dr*ci,self.dr*ci],'r-',linewidth=2)

        self.canvas.axes.set_yticks(ticklocs)
        self.canvas.axes.set_yticklabels(disp_chan_names)
        self.canvas.axes.set_ylim(-self.dr/2, (self.disp_chans_num - 1) * self.dr + self.dr/2)
        self.canvas.axes.set_xlabel('time(s)')
        self.canvas.draw()


    def disp_refresh_HFOdets_raw(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_ylim(self.y0, self.y1)
        segs = []
        ticklocs = []
        self.disp_start = int(self.disp_time_start * self.fs)
        self.disp_end = int((self.disp_time_start + self.disp_time_win) * self.fs)
        self.modRaw_edf_dataIndex=[self.origin_chans.index(x) for x in self.disp_ch_names]
        self.modRaw_edf_data=self.origin_data[self.modRaw_edf_dataIndex]
        self.modRaw_edf_data=self.modRaw_edf_data-np.mean(self.modRaw_edf_data,axis=0,keepdims=True)
        self.disp_end = min(self.disp_end, self.modRaw_edf_data.shape[1])
        if self.disp_chans_num >= self.modRaw_edf_data.shape[0]:
            self.disp_chans_start = 0
            self.disp_chans_num = self.modRaw_edf_data.shape[0]
        elif self.disp_chans_start + self.disp_chans_num >= self.modRaw_edf_data.shape[0]:
            self.disp_chans_start = self.modRaw_edf_data.shape[0] - self.disp_chans_num
        for i in range(self.disp_chans_start, self.disp_chans_start + self.disp_chans_num):
            tmp_data = self.modRaw_edf_data[i, self.disp_start:self.disp_end]

            tmp_time = np.linspace(self.disp_start / self.fs, self.disp_end / self.fs, self.disp_end - self.disp_start)
            tmp_data = tmp_data * self.disp_wave_mul
            segs.append(np.hstack((tmp_time[:, np.newaxis], tmp_data[:, np.newaxis])))
            ticklocs.append((i - self.disp_chans_start) * self.dr)
        offsets = np.zeros((self.disp_chans_num, 2), dtype=float)
        offsets[:, 1] = ticklocs
        colors = self.edf_line_colors[self.disp_chans_start:self.disp_chans_start + self.disp_chans_num]
        # linewidths=
        lines = LineCollection(segs, offsets=offsets, linewidths=0.7,
                               transOffset=None,colors='k')  # ,colors=colors,transOffset=None)
        disp_chan_names = self.disp_ch_names[
                          self.disp_chans_start:(self.disp_chans_start + self.disp_chans_num)]
        self.canvas.axes.set_xlim(segs[0][0, 0], segs[0][-1, 0])
        self.canvas.axes.add_collection(lines)

        # add hfo detections
        self.hfoDets_resultsFile = os.path.join('./HFOdets',
                                                os.path.basename(self.mat_filename).split('.')[0] + '_events.npz')
        if os.path.exists(self.hfoDets_resultsFile):
            self.hfoDets = np.load(self.hfoDets_resultsFile, allow_pickle=True)
            self.hfoDets_chns = self.hfoDets['file_chnsNames'].tolist()
            self.hfoDets_times = self.hfoDets['file_highEvents_times']

        showDets_index = [self.hfoDets_chns.index(x) if x in self.hfoDets_chns else [] for x in disp_chan_names]
        showDets_times = [self.hfoDets_times[x] for x in showDets_index]
        for ci in range(len(showDets_times)):
            if len(showDets_times[ci]) == 0:
                continue
            for ti, tw in enumerate(showDets_times[ci]):
                if tw[0] > (self.disp_start / self.fs) and tw[1] < (self.disp_end / self.fs):
                    self.canvas.axes.plot([tw[0], tw[1]], [self.dr * ci, self.dr * ci], 'r-', linewidth=2)

        self.canvas.axes.set_yticks(ticklocs)
        self.canvas.axes.set_yticklabels(disp_chan_names)
        self.canvas.axes.set_ylim(-self.dr/2, (self.disp_chans_num - 1) * self.dr + self.dr/2)
        self.canvas.axes.set_xlabel('time(s)')
        # add first line
        self.canvas.draw()

#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = Brainquake()
#     sys.exit(app.exec_())
#
