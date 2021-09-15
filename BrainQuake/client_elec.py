#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import os
import re
import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

import PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QListWidget, QSizePolicy, QTableWidgetItem

# from Ui_form import Electrodes_gui
# import utils1 # mayaviplot
# from utils1 import Preprocess_thread, GenerateLabel_thread, PreprocessResult_thread, ContactSegment_thread
from gui_forms.elec_form import Electrodes_gui
from utils.elec_utils import Preprocess_thread, GenerateLabel_thread, PreprocessResult_thread, ContactSegment_thread, savenpy, lookupTable

SUBDIR = './SurfDataset'

### -------------- Electrode Module GUI Main Class -------------- ###
class Electrodes(QtWidgets.QWidget, Electrodes_gui):
    def __init__(self):
        super(Electrodes, self).__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.fig = Figure(figsize=(10,10))
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.c = ['#698E6A', '#896D47', '#D4BF89', '#106898', '#954024', '#A35F65',
                  '#535164', '#CDD171', '#BED2BB', '#4C1E1A', '#F5B087', '#CC5D20',
                  '#003460', '#ED6D46', '#822327', '#1E2732']
        self.thread_1 = Preprocess_thread()
        self.thread_1.finished.connect(self.preprocessFinished)
        self.thread_2 = GenerateLabel_thread()
        self.thread_2.finished.connect(self.genLabelFinished)
        self.thread_3 = PreprocessResult_thread()
        self.thread_3.send_axes.connect(self.preprocessView)
        # self.thread_4 = LabelResult_thread()
        # self.thread_4.finished.connect(self.genLabelFinished)
        self.thread_5 = ContactSegment_thread()
        self.thread_5.finished.connect(self.genContactFinished)
    
    def patientName(self):
        self.patient = self.lineEdit_1.text()

    def hospitalName(self):
        pass

    def importSurf(self):
        # select a freesurfer subject file folder
        self.directory_surf = QFileDialog.getExistingDirectory(self, "getExistingDirectory", os.getcwd())
        if not self.directory_surf:
            pass
        else: # a folder selected
            self.Patname_surf = self.directory_surf.split('/')[-1] # identify the patient name
            self.lineEdit_1.setText(self.Patname_surf) # set the patient name to lineEdit_1
            self.lineEdit_1.setReadOnly(True) # name set!
            self.patient = self.lineEdit_1.text() # save name to self
            self.pushButton_1.setEnabled(True) # release CT btn
            self.lineEdit_3.clear() # initialize the display
            self.lineEdit_3.setEnabled(False)
            self.lineEdit_4.clear()
            self.lineEdit_4.setEnabled(False)
            self.pushButton_3.setEnabled(False)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
            self.pushButton_7.setEnabled(False)
            self.pushButton_8.setEnabled(False)
            self.pushButton_9.setEnabled(False)
            self.pushButton_10.setEnabled(False)
            self.doubleSpinBox_1.setValue(0)
            self.doubleSpinBox_1.setEnabled(False)
            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(0)
            
            self.fig = Figure(figsize=(10,10))
            self.axes.clear()
            self.scene.addWidget(FigureCanvas(self.fig))
            self.graphicsView.show()

    def importCT(self):
        # import xxxCT_Reg.nii.gz file
        self.directory_ct, _ = QFileDialog.getOpenFileName(self, "getOpenFileName", "", "All Files (*);;Nifti Files (*.nii.gz)")
        if not self.directory_ct:
            pass
        else: # a xxxCT_Reg.nii.gz file selected
            self.Filename_ct = self.directory_ct.split('/')[-1] # identify ct filename
            self.Patname_ct = self.Filename_ct.split('.')[0].split('C')[0] # identify ct patient name
            self.directory_ct = self.directory_ct.split(self.Filename_ct)[0] # save ct directory without filename
            # 此处应有判等/警告和break
    
            self.lineEdit_3.setEnabled(True) # enable 3 parameters' input
            self.lineEdit_4.setEnabled(True)
            self.doubleSpinBox_1.setEnabled(True)
            
            find_flag = 0 # check for xxxCT_intracranial_thre_K_ero.nii.gz
            find_flag1 = 0 # check for xxx_labels.npy
            for root, dirs, files in os.walk(self.directory_ct):
                for filename in files:
                    if re.search(r'_intracranial_', filename): # if found preprocessed!
                        self.CTintra_file = os.path.join(self.directory_ct, filename) # save intra_file dir
                        self.ero_itr = int(filename.split('_')[-1].split('.')[0]) # read 3 parameters from filename
                        self.K = int(filename.split('_')[-2])
                        self.thre = float(filename.split('_')[-3])
                        
                        self.pushButton_3.setEnabled(True) # enable preprocess btn
                        self.pushButton_4.setEnabled(True) # enable preprocessView btn
                        self.lineEdit_3.setText(str(self.K)) # display 3 parameters
                        self.lineEdit_4.setText(str(self.ero_itr))
                        self.doubleSpinBox_1.setValue(self.thre*100)
                        find_flag = 1
                        break
            for root, dirs, files in os.walk(self.directory_ct):
                for filename in files:
                    if re.search(r'_labels.npy', filename): # if found preprocessed!
                        self.directory_labels = os.path.join(self.directory_ct, filename) # save intra_file dir
                        self.pushButton_6.setEnabled(True)
                        find_flag1 = 1
                        break        
                    
            if not find_flag: # if not found preprocessed!
                self.lineEdit_4.setText(str(10)) # set 2 parameters recommended, without variable K
                self.doubleSpinBox_1.setValue(10)
            
    def preprocessData(self):
        self.pushButton_4.setEnabled(False) # disable preoprocessView btn before thread ends
        
        for root, dirs, files in os.walk(self.directory_ct): # delete former intra_file
            for filename in files:
                if re.search(r'_intracranial', filename): 
                    os.remove(os.path.join(self.directory_ct, filename))
                    break
        
        # 此处应判别整数
        self.ero_itr = int(self.lineEdit_4.text()) # read 3 parameters from screen
        self.K = int(self.lineEdit_3.text())
        self.thre = float(self.doubleSpinBox_1.value())

        self.thread_1.patient = self.patient # start preprocessing thread
        self.thread_1.directory_ct = self.directory_ct
        self.thread_1.directory_surf = self.directory_surf
        self.thread_1.K = self.K
        self.thread_1.thre = self.thre
        self.thread_1.ero_itr = self.ero_itr
        self.thread_1.start()
    
    def preprocessFinished(self):
        find_flag = 0 # check for xxxCT_intracranial_thre_K_ero.nii.gz
        for root, dirs, files in os.walk(self.directory_ct):
            for filename in files:
                if re.search(r'_intracranial_', filename): # if found preprocessed!
                    self.CTintra_file = os.path.join(self.directory_ct, filename) # save intra_file dir
                    self.ero_itr = int(filename.split('_')[-1].split('.')[0]) # read 3 parameters from filename
                    self.K = int(filename.split('_')[-2])
                    self.thre = float(filename.split('_')[-3])
                    
                    self.pushButton_4.setEnabled(True) # enable preprocessView btn
                    self.lineEdit_3.setText(str(self.K)) # display 3 parameters
                    self.lineEdit_4.setText(str(self.ero_itr))
                    self.doubleSpinBox_1.setValue(self.thre*100)
                    find_flag = 1
                    break
        if not find_flag:
            # 此处提出警告，预处理失败，没有intra文件！
            pass

    def viewIntra(self):
        find_flag = 0 # check for intra_file
        for root, dirs, files in os.walk(self.directory_ct):
            for filename in files:
                if re.search(r'_intracranial_', filename): # if found intra_file
                    self.CTintra_file = os.path.join(self.directory_ct, filename) # save intra_file dir
                    self.ero_itr = int(filename.split('_')[-1].split('.')[0]) # read 3 parameters from filename
                    self.K = int(filename.split('_')[-2])
                    self.thre = float(filename.split('_')[-3])
                    
                    self.lineEdit_3.setText(str(self.K)) # display 3 parameters
                    self.lineEdit_4.setText(str(self.ero_itr))
                    self.doubleSpinBox_1.setValue(self.thre*100)
                    find_flag = 1
                    break
        if not find_flag:
            # 此处提出警告，没有找到intra文件！
            pass
        else: # intra_file exists and is possible to be OK for following steps
            self.pushButton_5.setEnabled(True) # enable labelGen btn
            self.thread_3.patient = self.patient
            self.thread_3.thre = self.thre
            self.thread_3.CTintra_file = self.CTintra_file
            self.thread_3.start()
    
    def preprocessView(self, pointsArray):
        self.xs = pointsArray[:, 0]
        self.ys = pointsArray[:, 1]
        self.zs = pointsArray[:, 2]

        self.fig = Figure(figsize=(10,10))
        self.scene.addWidget(FigureCanvas(self.fig))
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_title(f"Preprocessed Electrodes, patient={self.patient}")
        self.axes.set_xlim(0, 256)
        self.axes.set_ylim(0, 256)
        self.axes.set_zlim(0, 256)
        self.axes.set_axis_off()
        self.axes.scatter(pointsArray[:,0], pointsArray[:,1], pointsArray[:,2], marker='.', c='blue') # plot the cluster of electrodes

        # self.scene.addWidget(FigureCanvas(self.fig))
        self.graphicsView.show()

    def numberK(self):
        self.pushButton_3.setEnabled(True) # if variable K is set, enable preprocess btn
    
    def numberEro(self):
        pass

    def threSel(self):
        pass

    def labelGen(self): # btn 5
        find_flag = 0 # still need to check for intra_file existence, since at this time preprocessing group is still editable
        for root, dirs, files in os.walk(self.directory_ct):
            for filename in files:
                if re.search(r'_intracranial_', filename):
                    self.CTintra_file = os.path.join(self.directory_ct, filename) # read from filename
                    self.ero_itr = int(filename.split('_')[-1].split('.')[0])
                    self.K = int(filename.split('_')[-2])
                    self.thre = float(filename.split('_')[-3])
                    
                    self.pushButton_1.setEnabled(False) # label btn clicked so disable the above processable btns
                    self.pushButton_3.setEnabled(False)
                    self.lineEdit_3.setText(str(self.K)) # label btn clicked so disable the above editable lines
                    self.lineEdit_4.setText(str(self.ero_itr))
                    self.doubleSpinBox_1.setValue(self.thre*100)
                    self.lineEdit_3.setEnabled(False)
                    self.lineEdit_4.setEnabled(False)
                    self.doubleSpinBox_1.setEnabled(False)
                    find_flag = 1
                    break
        if not find_flag:
            # 此处提出警告，没有找到intra文件！
            pass
        else: # intra_file exists!
            self.thread_2.patient = self.patient
            self.thread_2.directory_ct = self.directory_ct
            self.thread_2.intra_file = self.CTintra_file
            self.thread_2.K = self.K
            self.thread_2.start()

    def genLabelFinished(self, k_flag):
        if k_flag: 
            # 此处应警告，聚类K_check<K
            pass
        else:
            find_flag = 0 # check for label_file existance
            for root, dirs, files in os.walk(self.directory_ct):
                for filename in files:
                    if re.search(r'_labels.npy', filename):
                        self.pushButton_6.setEnabled(True) # enable labelView btn
                        self.directory_labels = os.path.join(self.directory_ct, filename)
                        find_flag = 1
                        break
            if not find_flag:
                pass
            else:
                self.labels = np.load(self.directory_labels, allow_pickle=True)
                self.fig = Figure(figsize=(10,10))
                self.scene.addWidget(FigureCanvas(self.fig))
                # self.fig.cla()
                self.axes.clear()
                self.axes = self.fig.add_subplot(111, projection='3d')
                self.axes.set_title(f"Clustered {self.K} Electrodes, patient={self.patient}")
                self.axes.set_xlim(0, 256)
                self.axes.set_ylim(0, 256)
                self.axes.set_zlim(0, 256)
                self.axes.set_axis_off()
                for i in range(self.K):
                    indx, indy, indz = np.where(self.labels == i+1)
                    self.axes.scatter(indx, indy, indz, marker='.', c=self.c[i])
        
                # self.scene.addWidget(FigureCanvas(self.fig))
                self.graphicsView.show()
                self.pushButton_8.setEnabled(True) # enable labelDone btn
            
    def viewLabels(self): # btn 6 empty
        self.pushButton_5.setEnabled(True) # enable label btn
        self.pushButton_8.setEnabled(True) # enable labelDone btn
        # self.pushButton_7.setEnabled(True) # enable contactSeg btn
        self.labels = np.load(self.directory_labels, allow_pickle=True)
        self.fig = Figure(figsize=(10,10))
        self.scene.addWidget(FigureCanvas(self.fig))
        self.axes.clear()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_title(f"Clustered {self.K} Electrodes, patient={self.patient}")
        self.axes.set_xlim(0, 256)
        self.axes.set_ylim(0, 256)
        self.axes.set_zlim(0, 256)
        self.axes.set_axis_off()
        for i in range(self.K):
            indx, indy, indz = np.where(self.labels == i+1)
            self.axes.scatter(indx, indy, indz, marker='.', c=self.c[i])

        # self.scene.addWidget(FigureCanvas(self.fig))
        self.graphicsView.show()
    
    def labelsDone(self): # btn 8
        self.pushButton_1.setEnabled(False) # disable CTimport btn
        self.pushButton_3.setEnabled(False) # diable prepro btn
        self.pushButton_5.setEnabled(False) # disable label btn
        self.pushButton_6.setEnabled(True) # enable contactSeg btn
        self.pushButton_7.setEnabled(True) # enable contactSeg btn

    def contactSeg(self): # btn 7
        self.pushButton_8.setEnabled(False) # disable labelDone btn
        self.thread_5.K = self.K
        self.thread_5.directory_labels = self.directory_ct
        self.thread_5.patName = self.patient
        self.thread_5.numMax = 16
        self.thread_5.diameterSize=2
        self.thread_5.spacing=3.5
        self.thread_5.gap=0
        self.thread_5.start()
    
    def genContactFinished(self):
        self.pushButton_9.setEnabled(True)
        
    def elecAdjust(self):
        pass

    def viewContacts(self):

        # utils.savenpy(filePath=self.directory_ct, patientName=self.patient)
        savenpy(filePath=self.directory_ct, patientName=self.patient)
        
        ## set tableWidget
        dir = f"{self.directory_ct}/{self.patient}_result"
        self.elec_dict = {}
        self.elec_number_dict = {}
        self.elec_label_dict = {}
        for root, dirs, files in os.walk(dir, topdown=True):
            if '.DS_Store' in files:
                files.remove('.DS_Store')
            for file in files:
                elec_name = file.split('.')[0]
                elec_info = np.loadtxt(os.path.join(root, file))
                elec_number = elec_info.shape[0]
                self.elec_dict[elec_name] = elec_info
                self.elec_number_dict[elec_name] = elec_number
        print(self.elec_dict)
        print(self.elec_number_dict)
        for item in self.elec_number_dict:
            row = self.tableWidget.rowCount()
            self.tableWidget.setRowCount(row + 1)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(item))
            number = str(self.elec_number_dict[item])
            self.tableWidget.setItem(row, 1, QTableWidgetItem(number))
            # labels_name = utils.elec_utils.lookupTable(subdir=SUBDIR, patient=self.patient, ctdir=self.directory_ct, elec_label=item)
            labels_name = lookupTable(subdir=SUBDIR, patient=self.patient, ctdir=self.directory_ct, elec_label=item)
            self.tableWidget.setItem(row, 2, QTableWidgetItem(labels_name[0]))
            self.elec_label_dict[item] = labels_name
        print(self.elec_label_dict)
        # mayaviplot.mayaviView(filePath=DATASETDIR, surfPath=f"{SUBDIR}/subjects", subname=self.patient)
        self.fig = Figure(figsize=(10,10))
        self.scene.addWidget(FigureCanvas(self.fig))
        self.axes.clear()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_title(f"Segmented contacts of {self.K} Electrodes, patient={self.patient}")
        self.axes.set_xlim(0, 256)
        self.axes.set_ylim(0, 256)
        self.axes.set_zlim(0, 256)
        self.axes.set_axis_off()
        for i in range(self.K):
            indx, indy, indz = np.where(self.labels == i+1)
            self.axes.scatter(indx, indy, indz, marker='.', c=self.c[i])
        for item in self.elec_dict:
            self.axes.scatter(128-self.elec_dict[item][:,0], 128-self.elec_dict[item][:,2], 128+self.elec_dict[item][:,1], marker='*', c='red')
            self.axes.text(130-self.elec_dict[item][0,0], 130-self.elec_dict[item][0,2], 130+self.elec_dict[item][0,1], f"{item}", c='black')
        # self.scene.addWidget(FigureCanvas(self.fig))
        self.graphicsView.show()

    def allSet(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = Electrodes()
    widget.showMaximized()
    widget.show()
    sys.exit(app.exec_())
