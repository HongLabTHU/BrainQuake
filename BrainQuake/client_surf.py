#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import socket
import time
import pickle
import os
import mayavi
from mayavi import mlab
import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QGraphicsScene

# from form import Ui_reconSurfer
# import utils_cs
from utils import surfer_utils
from gui_forms.surfer_form import Ui_reconSurfer

HEADERSIZE = 10
SEPARATOR = '<SEPARATOR>'
BUFFER_SIZE = 4096
host = '166.111.152.123'
port = 6669
Filepath = '.'

class Uploader(QThread):
    progressBarValue = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self):
        super(Uploader, self).__init__()
    
    def run(self):
        ## sending task_type 'Upload': '1x', '10' for recon-all & '11' for fast-surfer
        self.s1 = surfer_utils.create_socket(host, port)
        time.sleep(1)
        self.task_type = '1'
        self.task_type_all = self.task_type + self.reconType
        # surfer_utils.text_send(self.s1, self.task_type)
        surfer_utils.text_send(self.s1, self.task_type_all)
        self.filesize = os.path.getsize(self.patientFilepath)
        
        ## sending a T1 file
        time.sleep(1)
        self.s1.send(f'{self.fileName}{SEPARATOR}{self.filesize}'.encode())
        time.sleep(1)
        with open(self.patientFilepath, "rb") as f:
            j = 0
            while True:
                ## read the bytes from the file
                bytes_read = f.read(BUFFER_SIZE)
                j = j + 1
                i = int(100*j*BUFFER_SIZE/self.filesize)
                if not bytes_read: ## file transmitting is done
                    break
                ## we use sendall to assure transimission in busy networks
                self.s1.sendall(bytes_read)
                self.progressBarValue.emit(i)      
        f.close()
        self.s1.close()
        ## receive a msg
        time.sleep(3)
        self.s2 = surfer_utils.create_socket(host, 6666)
        log_read = surfer_utils.text_recv(self.s2)
        self.log.emit(log_read)
        self.s2.close()

class Checker(QThread):
    logs = pyqtSignal(str)

    def __init__(self):
        super(Checker, self).__init__()

    def run(self):
        ## sending task_type 'Check': '2'
        self.s1 = surfer_utils.create_socket(host, port)
        time.sleep(1)
        self.task_type = '2'
        surfer_utils.text_send(self.s1, self.task_type)
        self.s1.close()

        ## send a request to check
        time.sleep(1)
        self.s2 = surfer_utils.create_socket(host, 6665)
        time.sleep(1)
        if self.name == '<name>':
            self.name = 'None'
        if self.number == '<number>':
            self.number = 'None'
        self.req = 'client'
        self.state = 'None'
        self.info = 'None'
        self.check_log = ' '.join([self.number, self.name, self.hospital, self.state, self.info])
        # print(self.check_log)
        surfer_utils.text_send(self.s2, self.check_log)
        logs = surfer_utils.text_recv(self.s2)
        # print(logs)
        self.logs.emit(logs)
        self.s2.close()

class Downloader(QThread):
    downloadValue = pyqtSignal(str)

    def __init__(self):
        super(Downloader, self).__init__()

    def run(self):
        ## sending task_type 'Download': '3'
        self.s1 = surfer_utils.create_socket(host, port)
        time.sleep(1)
        self.task_type = '3'
        surfer_utils.text_send(self.s1, self.task_type)
        self.s1.close()
        
        ## receiving a recon zip file
        time.sleep(1)
        self.s2 = surfer_utils.create_socket(host, 6664)
        time.sleep(1)
        surfer_utils.text_send(self.s2, self.downloadlist)
        time.sleep(1)
        received = self.s2.recv(BUFFER_SIZE).decode()
        filename, filesize = received.split(SEPARATOR)
        filename = os.path.basename(filename) ## remove absolute path if there is
        filepath = os.path.join(Filepath, 'data', 'down', filename)
        filesize = int(filesize) ## convert to integer
        time.sleep(1)
        ## start receiving the file from the socket and writing to the file stream
        with open(filepath, "wb") as f:
            j = 0
            while True:
                ## read bytes from the socket (receive)
                bytes_read = self.s2.recv(BUFFER_SIZE)
                j = j + 1
                i = [99, int(100*j*BUFFER_SIZE/filesize)][int(100*j*BUFFER_SIZE/filesize)<99]
                if not bytes_read:    
                    ## nothing is received
                    ## file transmitting is done
                    ## print(f"transmission completed")
                    break
                ## write to the file the bytes we just received
                f.write(bytes_read)
                self.downloadValue.emit(f"{i}%")
        # surfer_utils.file_recv(self.s2)
        # print('recvd!')
        info = 'Done!'
        self.downloadValue.emit(info) 
        self.s2.close()

# class Previewer(FigureCanvas):
#     def __init__(self, width=5, height=4, dpi=100):
#         self.fig = Figure(figsize=(width, height), dpi=dpi)
#         super(Previewer, self).__init__(self.fig)
#         self.axes = self.fig.add_subplot(111, projection='3d')
    
#     def mayaviplot(self, name):
#         self.previewlist = name
#         print(self.previewlist)
#         self.zipfilepath = os.path.join(Filepath, 'data', 'down', f"{self.previewlist}.zip")
#         if os.path.isfile(self.zipfilepath):
#             # os.system(f"unzip {self.zipfilepath}")
#             self.pialfilepath = os.path.join(Filepath, 'data', 'down', self.previewlist, 'surf')
#         lh_pial_file=os.path.join(self.pialfilepath,'lh.pial')
#         rh_pial_file=os.path.join(self.pialfilepath,'rh.pial')
#         verl,facel=nib.freesurfer.read_geometry(lh_pial_file)
#         verr,facer=nib.freesurfer.read_geometry(rh_pial_file)
#         verall=np.concatenate([verl,verr],axis=0)
#         facer=facer+verl.shape[0]
#         faceall=np.concatenate([facel,facer],axis=0)
#         print(verall[:,0])
#         print(faceall)
#         # mlab.draw()
#         mlab.triangular_mesh(verall[:,0], verall[:,1], verall[:,2], faceall, color=(1,1,1), opacity=0.5)
#         mlab.draw()
        

class reconSurferUi(QtWidgets.QWidget, Ui_reconSurfer):
    def __init__(self):
        super(reconSurferUi, self).__init__()
        self.setupUi(self)
        self.namelist = []
        self.numberlist = []
        self.checklist = []
        self.thread_1 = Uploader()
        self.thread_1.progressBarValue.connect(self.progressValue)
        self.thread_1.log.connect(self.progressLog)
        self.thread_2 = Checker()
        self.thread_2.logs.connect(self.logsPreview)
        self.thread_3 = Downloader()
        self.thread_3.downloadValue.connect(self.downloadProgress)
        # self.thread_4 = Previewer()

    def browseT1File(self):
        self.directory = QFileDialog.getOpenFileName(self, \
             "getOpenFileName", "", "All Files (*);;Nifti Files (*.nii.gz)")
        self.Filepath = self.directory[0]
        self.Filename = self.directory[0].split('/')[-1]
        self.Patname = self.directory[0].split('/')[-1].split('.')[0]
        self.textBrowser.setText(self.Filepath)
        self.textBrowser_2.setText(self.Filename)
        self.progressBar.setValue(0)

    def uploadT1File(self):
        self.reconType = self.comboBox.currentIndex()
        self.thread_1.patientName = self.Patname 
        self.thread_1.fileName = self.Filename
        self.thread_1.patientFilepath = self.Filepath 
        self.thread_1.reconType = str(self.reconType)
        self.thread_1.start()

    def checkProgress(self):
        self.thread_2.number = self.comboBox_3.currentText()
        self.thread_2.name = self.comboBox_2.currentText()
        self.thread_2.hospital = self.comboBox_4.currentText()
        self.thread_2.start()

    def downloadRecon(self):
        self.itemsSelected()
        for item in self.items:
            # assert str(item.split(' ')[-1]) == str(1), "The selected recon has not completed!"
            if str(item.split(' ')[-1]) == str(1):
                self.thread_3.downloadlist = self.items
                self.thread_3.start()
            else:
                pass
        # if ~(self.thread_3.isRunning()): ###Bug!!!
        # self.thread_3.downloadlist = self.items
        # self.thread_3.start()
    
    def previewRecon(self):
        self.itemsSelected()
        # print(self.items)
        self.item = self.items[0]
        # print(self.item)
        self.logSel = self.item.split(' ')[1]
        # print(self.logSel)
        self.mayaviplot(name=self.logSel)
        # self.F = Previewer(width=7, height=6, dpi=100)
        # self.thread_4.previewlist = self.logSel
        # self.thread_4.start()
    
    def mayaviplot(self, name):
        self.previewlist = name
        print(self.previewlist)
        self.zipfilepath = os.path.join(Filepath, 'data', 'down', f"{self.previewlist}.zip")
        self.unzipfilepath = os.path.join(Filepath, 'data', 'down', f"{self.previewlist}")
        if os.path.exists(self.zipfilepath):
            if not os.path.exists(self.unzipfilepath):
                os.system(f"unzip {self.zipfilepath} -d ./data/down")
            self.pialfilepath = os.path.join(Filepath, 'data', 'down', self.previewlist, 'surf')
            lh_pial_file=os.path.join(self.pialfilepath,'lh.pial')
            rh_pial_file=os.path.join(self.pialfilepath,'rh.pial')
            verl,facel=nib.freesurfer.read_geometry(lh_pial_file)
            verr,facer=nib.freesurfer.read_geometry(rh_pial_file)
            verall=np.concatenate([verl,verr],axis=0)
            facer=facer+verl.shape[0]
            faceall=np.concatenate([facel,facer],axis=0)
            # print(verall[:,0])
            # print(faceall)
            mlab.triangular_mesh(verall[:,0], verall[:,1], verall[:,2], faceall)
            mlab.draw()
        else:
            pass

    def itemsSelected(self):
        items = self.tableWidget.selectedItems()
        indexes = self.tableWidget.selectedIndexes()
        self.items = []
        self.indexes = []
        for item in items:
            item = item.text()
            self.items.append(item)
        for index in indexes:
            index = index.row()
            self.indexes.append(index)
        
    def progressValue(self, i):
        self.progressBar.setValue(i)

    def progressLog(self, log_read):
        self.textBrowser_2.setText(log_read)
        number = log_read.split(' ')[0]
        name = log_read.split(' ')[1]
        self.numberlist.append(number)
        self.comboBox_3.addItems(self.numberlist)
        self.namelist.append(name)
        self.comboBox_2.addItems(self.namelist)
    
    def logsPreview(self, logs):
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        self.checklist = logs.split("\n")
        if self.checklist[-1] == '':
            self.checklist.remove(self.checklist[-1])
        i = 0
        for log in self.checklist:
            row = self.tableWidget.rowCount()
            self.tableWidget.setRowCount(row + 1)
            table_item = str(self.checklist[i])
            newItem = QTableWidgetItem(table_item) 
            self.tableWidget.setItem(row, 0, newItem)
            i = i + 1
    
    def downloadProgress(self, info):
        row = self.indexes[0]
        self.tableWidget.setItem(row, 1, QTableWidgetItem(info))

#    def closeEvent(self, event):
#        reply = QtWidgets.QMessageBox.question(self,'reconSurfer',"Quit?", \
#            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
#        if reply == QtWidgets.QMessageBox.Yes:
#            event.accept()
#            os._exit(0)
#        else:
#            event.ignore()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = reconSurferUi()
    widget.setFixedSize(860, 640)
    widget.show()
    sys.exit(app.exec_())
