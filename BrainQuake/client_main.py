#! /usr/bin/python3.6
# -- coding: utf-8 -- **

import sys

from PyQt5.QtWidgets import QApplication,  QMainWindow, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLineEdit, QDesktopWidget, QGridLayout, QFileDialog,  QListWidget, QLabel,QFrame,QGroupBox
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont,QPixmap

from client_ictal import IctalModule
from client_inter import InterModule
from client_elec import Electrodes
from client_surf import reconSurferUi


class quakeMain(QWidget):
    def __init__(self):
        super(quakeMain,self).__init__()
        self.init_gui()

    def init_gui(self):
        self.setWindowTitle('BrainQuake')
        self.resize(500,300)
        self.centerWin()
        self.setStyleSheet('background-color:lightgrey;')
        self.setAttribute(Qt.WA_MacShowFocusRect,0)
        self.gridlayout=QGridLayout()

        #pre setting
        self.button_Adaptive = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.button_font=QFont()
        self.button_font.setFamily("Arial")
        self.button_font.setPointSize(18)

        self.mainLabel=QLabel(self)
        self.mainLabel.setPixmap(QPixmap('../docs/round_icon_min.png'))
        self.mainLabel.setAlignment(Qt.AlignCenter)
        self.gridlayout.addWidget(self.mainLabel,1,2,1,1)

        self.mainWords=QLabel(self)
        self.mainWords_font=QFont()
        self.mainWords_font.setFamily('black')
        self.mainWords_font.setPointSize(35)
        self.mainWords_font.setBold(True)
        self.mainWords.setText('BrainQuake')
        self.mainWords.setFont(self.mainWords_font)
        self.gridlayout.addWidget(self.mainWords,1,3,1,1)

        self.frame=QGroupBox(self)
        # self.frame.resize(300,200)
        self.frame.setStyleSheet("QGroupBox{border: 2px solid gray; border-radius: 5px;background-color:lightgrey;}QGroupBox:title{subcontrol-origin: margin;subcontrol-position: top left;padding: 0 3px 0 3px;}")
        self.frame.setTitle('Computation Functions')
        self.frame.setFont(self.button_font)


        self.gridlayout.addWidget(self.frame,3,1,4,4)
        self.frame_layout=QGridLayout()


        # electrodes extraction
        self.button_elecs = QPushButton('Electrodes extraction', self)
        self.button_elecs.setToolTip('extract seeg electrodes locations')
        self.button_elecs.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:dimgrey;}QPushButton:hover{background-color:k}")
        self.button_elecs.setFont(self.button_font)
        self.frame_layout.addWidget(self.button_elecs, 1, 1, 2, 2)
        self.button_elecs.setSizePolicy(self.button_Adaptive)
        self.button_elecs.clicked.connect(self.elecs_computation)

        # surface reconstruction
        self.button_surfs = QPushButton('Surface reconstruction', self)
        self.button_surfs.setToolTip('pial surface reconstruction')
        self.button_surfs.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:dimgrey;}QPushButton:hover{background-color:k}")
        self.button_surfs.setFont(self.button_font)
        self.frame_layout.addWidget(self.button_surfs, 1, 3, 2, 2)
        self.button_surfs.setSizePolicy(self.button_Adaptive)
        self.button_surfs.clicked.connect(self.surfs_computation)

        # ictal module
        self.button_ictal=QPushButton('Ictal module', self)
        self.button_ictal.setToolTip('compute epilepsy index(EI) & full band characteristic(Full Band)')
        self.button_ictal.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:dimgrey;}QPushButton:hover{background-color:k}")
        self.button_ictal.setFont(self.button_font)
        self.frame_layout.addWidget(self.button_ictal,3,1,2,2)
        self.button_ictal.setSizePolicy(self.button_Adaptive)
        self.button_ictal.clicked.connect(self.ictal_computation)

        # interictal module
        self.button_inter = QPushButton('Interictal module', self)
        self.button_inter.setToolTip('compute high frequency events index(HI)')
        self.button_inter.setStyleSheet("QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:dimgrey;}QPushButton:hover{background-color:k}")
        self.button_inter.setFont(self.button_font)
        self.frame_layout.addWidget(self.button_inter, 3, 3, 2, 2)
        self.button_inter.setSizePolicy(self.button_Adaptive)
        self.button_inter.clicked.connect(self.inter_computation)


        self.frame.setLayout(self.frame_layout)
        self.setLayout(self.gridlayout)
        self.show()

    def centerWin(self):
        qr=self.frameGeometry()
        DeskCenter=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(DeskCenter)
        self.move(qr.topLeft())

    def ictal_computation(self):
        self.ictal_widget=IctalModule(self)
        self.ictal_widget.show()

    def inter_computation(self):
        self.inter_widget=InterModule(self)
        self.inter_widget.show()

    def elecs_computation(self):
        self.elec_widget=Electrodes()
        self.elec_widget.show()

    def surfs_computation(self):
        pass
        # self.surf_widget=reconSurferUi()
        # self.surf_widget.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = quakeMain()
    sys.exit(app.exec_())
