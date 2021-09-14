# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form(2).ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,  QMainWindow, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLineEdit, QDesktopWidget, QGridLayout, QFileDialog,  QListWidget, QLabel,QFrame,QGroupBox, QTableWidget
from PyQt5.QtCore import Qt, QThread
#from PyQt5.QtGui import QFont

class Ui_reconSurfer(object):
    def setupUi(self, reconSurferUi):
        reconSurferUi.setObjectName("reconSurferUi")
        #reconSurferUi.resize(630, 390)
        self.setStyleSheet('background-color:lightgrey;')
        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.setSpacing(20)
        self.gridlayout.setContentsMargins(35, 35, 35, 35)
        self.gridlayout.setColumnMinimumWidth(0, 30)
        self.gridlayout.setColumnMinimumWidth(1, 120)
        self.gridlayout.setColumnMinimumWidth(2, 30)
        self.gridlayout.setColumnMinimumWidth(3, 120)
        self.gridlayout.setColumnMinimumWidth(4, 30)
        self.gridlayout.setColumnMinimumWidth(5, 120)
        self.gridlayout.setColumnMinimumWidth(6, 30)
        self.gridlayout.setColumnMinimumWidth(7, 50)
        self.gridlayout.setColumnMinimumWidth(8, 30)
        self.gridlayout.setRowMinimumHeight(0, 10)
        self.gridlayout.setRowMinimumHeight(1, 10)
        self.gridlayout.setRowMinimumHeight(2, 10)
        self.gridlayout.setRowMinimumHeight(3, 10)
        self.gridlayout.setRowMinimumHeight(4, 10)
        self.gridlayout.setRowMinimumHeight(5, 10)
        self.gridlayout.setRowMinimumHeight(6, 10)
        self.gridlayout.setRowMinimumHeight(7, 10)
        self.gridlayout.setRowMinimumHeight(8, 80)
        self.gridlayout.setRowMinimumHeight(9, 80)
        self.gridlayout.setRowMinimumHeight(10, 80)
        #input group
        self.groupBox = QtWidgets.QGroupBox(reconSurferUi)
        self.groupBox.setObjectName("CreateANewTask")
        self.groupBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.groupBox.setStyleSheet("QGroupBox{border: 2px solid gray; border-radius: 5px; \
            background-color:lightgrey;}QGroupBox:title{subcontrol-origin: margin; \
            subcontrol-position: top left;padding: 0 3px 0 3px;}")
        self.gridlayout.addWidget(self.groupBox, 0, 0, 5, 9)
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setObjectName("FastOrReconall")
        self.comboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.comboBox.setStyleSheet("QComboBox{border-radius:5px;padding:5px; \
            background-color:dimgrey;color:#ffffff;}QComboBox:drop-down{border-top-right-radius:5px; \
            border-bottom-right-radius:5px;border-left-color:darkgrey;border-left-style:solid;}")
        self.comboBox.setFixedWidth(120)
        self.comboBox.setFixedHeight(30)
        self.comboBox.addItem("recon-all")
        self.comboBox.addItem("fast-surfer")
        self.gridlayout.addWidget(self.comboBox, 1, 1, 1, 1)
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setObjectName("T1FilePath")
        self.textBrowser.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.textBrowser.setStyleSheet("QTextBrowser{border-radius:5px;padding:5px;background-color:#ffffff;}")
        self.textBrowser.setFixedHeight(40)
        self.gridlayout.addWidget(self.textBrowser, 2, 1, 1, 5)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName("Browse")
        self.pushButton.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff; \
            background-color:dimgrey;}QPushButton:hover{background-color:k;}")
        self.pushButton.setText('Browse')
        self.pushButton.setToolTip('choose a T1 file')
        self.gridlayout.addWidget(self.pushButton, 2, 7, 1, 1)
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_2.setObjectName("T1PatientName")
        self.textBrowser_2.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.textBrowser_2.setStyleSheet("QTextBrowser{border-radius:5px;padding:5px;background-color:#ffffff;}")
        self.textBrowser_2.setFixedHeight(40)
        self.textBrowser_2.setFixedWidth(300)
        self.gridlayout.addWidget(self.textBrowser_2, 3, 1, 1, 3)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.gridlayout.addWidget(self.progressBar, 3, 4, 1, 2)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("Upload")
        self.pushButton_2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton_2.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff; \
            background-color:dimgrey;}QPushButton:hover{background-color:k;}")
        self.pushButton_2.setText('Upload')
        self.pushButton_2.setToolTip('upload a T1 file')
        self.gridlayout.addWidget(self.pushButton_2, 3, 7, 1, 1)
        
        self.groupBox_1 = QtWidgets.QGroupBox(reconSurferUi)
        self.groupBox_1.setObjectName("CheckTasks")
        self.groupBox_1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.groupBox_1.setStyleSheet("QGroupBox{border: 2px solid gray; border-radius: 5px; \
            background-color:lightgrey;}QGroupBox:title{subcontrol-origin: margin; \
            subcontrol-position: top left;padding: 0 3px 0 3px;}")
        self.gridlayout.addWidget(self.groupBox_1, 5, 0, 7, 9)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_1)
        self.comboBox_2.setObjectName("Name")
        self.comboBox_2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboBox_2.setStyleSheet("QComboBox{border-radius:5px;padding:5px; \
            background-color:dimgrey;color:#ffffff;}QComboBox:drop-down{border-top-right-radius:5px; \
            border-bottom-right-radius:5px;border-left-color:darkgrey;border-left-style:solid;}")
        #self.comboBox_2.setFixedWidth(120)
        self.comboBox_2.addItem("")
        self.gridlayout.addWidget(self.comboBox_2, 6, 1, 1, 1)
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_1)
        self.comboBox_3.setObjectName("Number")
        self.comboBox_3.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboBox_3.setStyleSheet("QComboBox{border-radius:5px;padding:5px; \
            background-color:dimgrey;color:#ffffff;}QComboBox:drop-down{border-top-right-radius:5px; \
            border-bottom-right-radius:5px;border-left-color:darkgrey;border-left-style:solid;}")
        self.comboBox_3.addItem("")
        self.gridlayout.addWidget(self.comboBox_3, 6, 3, 1, 1)
        self.comboBox_4 = QtWidgets.QComboBox(self.groupBox_1)
        self.comboBox_4.setObjectName("Hospital")
        self.comboBox_4.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboBox_4.setStyleSheet("QComboBox{border-radius:5px;padding:5px; \
            background-color:dimgrey;color:#ffffff;}QComboBox:drop-down{border-top-right-radius:5px; \
            border-bottom-right-radius:5px;border-left-color:darkgrey;border-left-style:solid;}")
        self.comboBox_4.addItem("")
        self.gridlayout.addWidget(self.comboBox_4, 6, 5, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_3.setObjectName("Check")
        self.pushButton_3.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton_3.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff; \
            background-color:dimgrey;}QPushButton:hover{background-color:k;}")
        self.pushButton_3.setText('Check')
        self.pushButton_3.setToolTip('check for recon processes')
        self.gridlayout.addWidget(self.pushButton_3, 6, 7, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_4.setObjectName("Download")
        self.pushButton_4.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton_4.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff; \
            background-color:dimgrey;}QPushButton:hover{background-color:k;}")
        self.pushButton_4.setText('Download')
        self.pushButton_4.setToolTip('download a recon zip file')
        self.gridlayout.addWidget(self.pushButton_4, 7, 7, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_1)
        self.pushButton_5.setObjectName("Preview")
        self.pushButton_5.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton_5.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff; \
            background-color:dimgrey;}QPushButton:hover{background-color:k;}")
        self.pushButton_5.setText('Preview')
        self.pushButton_5.setToolTip('preview a recon surface')
        self.gridlayout.addWidget(self.pushButton_5, 8, 7, 1, 1)
        # self.listWidget = QtWidgets.QListWidget(self.groupBox_1)
        # self.listWidget.setObjectName("CheckList")
        # self.listWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.listWidget.setStyleSheet("border-radius:5px;padding:5px;background-color:#ffffff;")
        # self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        # self.gridlayout.addWidget(self.listWidget, 7, 1, 4, 5)
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_1)
        self.tableWidget.setObjectName("CheckList")
        self.tableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setColumnWidth(0, 400)
        self.tableWidget.setColumnWidth(1, 100)
        self.tableWidget.setHorizontalHeaderLabels(['Logs','Download'])
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.horizontalHeader().setSectionsClickable(False)
        self.tableWidget.setStyleSheet("QTableView{border-radius:5px;background-color:#ffffff;color:black;selection-background-color:dimgrey} \
            QHeaderView:section{background-color:dimgrey;color:white;border: 5px solid#6c6c6c; \
            border-top-right-radius:5px;border-top-left-radius:5px;}")
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gridlayout.addWidget(self.tableWidget, 7, 1, 4, 5)
        reconSurferUi.setLayout(self.gridlayout)

        self.retranslateUi(reconSurferUi)
        self.pushButton.clicked.connect(reconSurferUi.browseT1File)
        self.pushButton_2.clicked.connect(reconSurferUi.uploadT1File)
        self.comboBox_2.setEditable(True)
        self.comboBox_3.setEditable(True)
        self.pushButton_3.clicked.connect(reconSurferUi.checkProgress)
        self.pushButton_4.clicked.connect(reconSurferUi.downloadRecon)
        self.pushButton_5.clicked.connect(reconSurferUi.previewRecon)
        self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        QtCore.QMetaObject.connectSlotsByName(reconSurferUi)
    
    def retranslateUi(self, reconSurferUi):
        _translate = QtCore.QCoreApplication.translate
        reconSurferUi.setWindowTitle(_translate("reconSurferUi", "Surface Module"))
#        self.groupBox.setTitle(_translate("reconSurferUi", "创建新项目"))
#        self.groupBox_1.setTitle(_translate("reconSurferUi", "查询项目"))
        # self.comboBox.setItemText(0, _translate("reconSurferUi", "重建类型"))
        self.comboBox_2.setItemText(0, _translate("BrainQuake_v3", "<name>"))
        self.comboBox_3.setItemText(0, _translate("BrainQuake_v3", "<number>"))
        self.comboBox_4.setItemText(0, _translate("BrainQuake_v3", "Yuquan"))
#        self.pushButton.setText(_translate("reconSurferUi", "浏览"))
#        self.pushButton_2.setText(_translate("reconSurferUi", "上传"))
#        self.pushButton_3.setText(_translate("reconSurferUi", "查询"))
#        self.pushButton_4.setText(_translate("reconSurferUi", "下载"))
#        self.pushButton_5.setText(_translate("reconSurferUi", "预览"))
