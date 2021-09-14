#! /usr/bin/python3.6
# -- coding: utf-8 -- **

### Here are a set of functions used in elec_pipe
### and a set of qthread class for elec_main_gui

import os
import re
import math
import numpy as np
from numpy import ndarray
import nibabel as nib
from scipy import ndimage
from sklearn.mixture import GaussianMixture as GMM
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
#from mayavi import mlab
import electrode1

CMD_Hough3D = '/Users/fangcai/hough-3d-lines/hough3dlines'

def run(cmd):
    """
    Print the command.
    Execute a command string on the shell (on bash).
    
    Parameters
    ----------
    cmd : str
        Command to be sent to the shell.
    """
    print(f"Running shell command: {cmd}")
    os.system(cmd)
    print(f"Done!\n")

def align(inp, ref, xfm=None, out=None, dof=12, searchrad=True, bins=256, interp=None, cost="mutualinfo", sch=None, wmseg=None, init=None, finesearch=None,):
    """Aligns two images using FSLs flirt function and stores the transform between them
    Parameters
    ----------
    inp : str
        path to input image being altered to align with the reference image as a nifti image file
    ref : str
        path to reference image being aligned to as a nifti image file
    xfm : str, optional
        where to save the 4x4 affine matrix containing the transform between two images, by default None
    out : str, optional
        determines whether the image will be automatically aligned and where the resulting image will be saved, by default None
    dof : int, optional
        the number of degrees of free dome of the alignment, by default 12
    searchrad : bool, optional
        whether to use the predefined searchradius parameter (180 degree sweep in x, y, and z), by default True
    bins : int, optional
        number of histogram bins, by default 256
    interp : str, optional
        interpolation method to be used (trilinear,nearestneighbour,sinc,spline), by default None
    cost : str, optional
        cost function to be used in alignment (mutualinfo, corratio, normcorr, normmi, leastsq, labeldiff, or bbr), by default "mutualinfo"
    sch : str, optional
        the optional FLIRT schedule, by default None
    wmseg : str, optional
        an optional white-matter segmentation for bbr, by default None
    init : str, optional
        an initial guess of an alignment in the form of the path to a matrix file, by default None
    finesearch : int, optional
        angle in degrees, by default None
    """

    cmd = f"flirt -in {inp} -ref {ref}"
    if xfm is not None:
        cmd += f" -omat {xfm}"
    if out is not None:
        cmd += f" -out {out}"
    if dof is not None:
        cmd += f" -dof {dof}"
    if bins is not None:
        cmd += f" -bins {bins}"
    if interp is not None:
        cmd += f" -interp {interp}"
    if cost is not None:
        cmd += f" -cost {cost}"
    if searchrad is not None:
        cmd += " -searchrx -180 180 -searchry -180 180 " + "-searchrz -180 180"
    if sch is not None:
        cmd += f" -schedule {sch}"
    if wmseg is not None:
        cmd += f" -wmseg {wmseg}"
    if init is not None:
        cmd += f" -init {init}"
    run(cmd)

def align_nonlinear(inp, ref, xfm, out, warp, ref_mask=None, in_mask=None, config=None):
    """Aligns two images using nonlinear methods and stores the transform between them using fnirt
    Parameters
    ----------
    inp : str
        path to the input image
    ref : str
        path to the reference image that the input will be aligned to
    xfm : str
        path to the file containing the affine transform matrix created by align()
    out : str
        path for the desired output image
    warp : str
        the path to store the output file containing the nonlinear warp coefficients/fields
    ref_mask : str, optional
        path to the reference image brain_mask, by default None
    in_mask : str, optional
        path for the file with mask in input image space, by default None
    config : str, optional
        path to the config file specifying command line arguments, by default None
    """

    cmd = f"fnirt --in={inp} --ref={ref} --aff={xfm} --iout={out} --cout={warp} --warpres=8,8,8"
    if ref_mask is not None:
        cmd += f" --refmask={ref_mask} --applyrefmask=1"
    if in_mask is not None:
        cmd += f" --inmask={in_mask} --applyinmask=1"
    if config is not None:
        cmd += f" --config={config}"
    run(cmd)

def dataExtraction(intraFile, thre=0.2):
    rawData = nib.load(intraFile).get_fdata()
    maxVal = np.amax(rawData)
    # print(f"maxVal={maxVal}")
    thre = maxVal * thre
    threData = np.copy(rawData)
    threData[threData < thre] = 0
    xs, ys, zs = np.where(threData != 0)
    return xs, ys, zs

def trackRecognition(patient, cmd_hough3d, CTresult_dir, intraFile, thre=0.2):
    
    xs, ys, zs = dataExtraction(intraFile, thre)
    
    X = np.transpose(np.array((xs, ys, zs)))
    # print(X.shape)
    fname = f"{CTresult_dir}/{patient}_3dPointClouds.dat"
    np.savetxt(fname, X, fmt='%.4f', delimiter=',', newline='\n', header='point clouds', footer='', comments='# ', encoding=None)
    
    cmd_hough = f"{cmd_hough3d} -o {CTresult_dir}/{patient}.txt -minvotes 5 {fname}"
    run(cmd=cmd_hough)
    return xs, ys, zs

def locateLine(row, info):
    ax = info[row][1]
    ay = info[row][2]
    az = info[row][3]
    bx = info[row][4]
    by = info[row][5]
    bz = info[row][6]
    axx = np.linspace(ax, ax+bx*50, 50)
    ayy = np.linspace(ay, ay+by*50, 50)
    azz = np.linspace(az, az+bz*50, 50)
    return axx, ayy, azz

class Preprocess_thread(QThread):

    finished = pyqtSignal()

    def __init__(self):
        super(Preprocess_thread, self).__init__()

    def run(self): # erode, skull, intra_save
        mask_file = os.path.join(f"{self.directory_surf}/mri", f"mask.mgz")
        img_mask = nib.load(mask_file)
        data_mask = img_mask.get_fdata()
        data_mask_ero = ndimage.morphology.binary_erosion(data_mask, iterations=self.ero_itr)
        
        CTreg_file = os.path.join(self.directory_ct, f"{self.patient}CT_Reg.nii.gz")
        img_ct = nib.load(CTreg_file)
        data_ct = img_ct.get_fdata()
        maxVal = np.amax(data_ct)
        self.thre = self.thre / 100
        thre = maxVal * self.thre
        
        data_ct[data_mask_ero == 0] = 0
        img1 = nib.Nifti1Image(data_ct, img_ct.affine)
        intra_file1 = os.path.join(self.directory_ct, f"{self.patient}CT_intra.nii.gz")
        nib.save(img1, intra_file1)
        
        data_ct[data_ct < thre] = 0

        img0 = nib.Nifti1Image(data_ct, img_ct.affine)
        intra_file = os.path.join(self.directory_ct, f"{self.patient}CT_intracranial_{self.thre}_{self.K}_{self.ero_itr}.nii.gz")
        nib.save(img0, intra_file)
        self.finished.emit()

class PreprocessResult_thread(QThread):

    send_axes = pyqtSignal(ndarray)

    def __init__(self):
        super(PreprocessResult_thread, self).__init__()

    def run(self):
        intra_file = self.CTintra_file
        xs, ys, zs = dataExtraction(intraFile=intra_file, thre=0)
        pointsArray = np.transpose(np.vstack((xs, ys, zs)))
        self.send_axes.emit(pointsArray)

class GenerateLabel_thread(QThread):

    finished = pyqtSignal(int)

    def __init__(self):
        super(GenerateLabel_thread, self).__init__()

    def run(self):
        # process 3d line hough transform
        hough_file = f"{self.directory_ct}/{self.patient}.txt"
        if not os.path.exists(hough_file):
            xs, ys, zs = trackRecognition(patient=self.patient, cmd_hough3d=CMD_Hough3D, CTresult_dir=self.directory_ct, intraFile=self.intra_file, thre=0)
        else: # temporarily
            # xs, ys, zs = utils.trackRecognition(patient=patient, cmd_hough3d=CMD_Hough3D, CTresult_dir=CTresult_dir, intraFile=intra_file, thre=Thre)
            xs, ys, zs = dataExtraction(intraFile=self.intra_file, thre=0)
            pass
        
        # read detected lines' info
        elec_track = []
        with open(hough_file, 'r') as f:
            for line in f.readlines():
                a = re.findall(r"\d+\.?\d*", line)
                for i in range(len(a)):
                    a[i] = float(a[i])
                elec_track.append(a)
        # print(f"{len(elec_track)} tracks has been detected!\n")
        # print(elec_track)
        elec_track = np.array(elec_track)
        K_check = elec_track.shape[0]
        if K_check < self.K:
            self.finished.emit(1)
        else: # if K_check != K:
            print(f"Warning: {self.K} electrodes implanted, but {K_check} has been clustered by Hough!")
            # sys.exit()
        
            # process a gaussian mixture model for bug fixing
            centroids = np.array(elec_track[0:self.K, 1:4])
            # print(centroids)
            X = np.transpose(np.vstack((xs, ys, zs)))
            gmm = GMM(n_components=self.K, covariance_type='full',means_init=centroids, random_state=None).fit(X)
            labels = gmm.predict(X)
            # print(labels)
    
            Labels = np.zeros((256, 256, 256)) # labeled space
            for i in range(self.K):
                ind = np.where(labels == i)
                Labels[xs[ind], ys[ind], zs[ind]] = i + 1
            np.save(os.path.join(self.directory_ct, f"{self.patient}_labels.npy"), Labels, allow_pickle=True, fix_imports=True)
            self.finished.emit(0)

# class LabelResult_thread(QThread):
#     def __init__(self):
#         super(LabelResult_thread, self).__init__()

#     def run(self):
#         print('Yaah!')

class ContactSegment_thread(QThread):

    finished = pyqtSignal()

    def __init__(self):
        super(ContactSegment_thread, self).__init__()

    def run(self):
        print('Yaah!')
        for i in range(self.K):
            iLabel = i + 1
            xxx = electrode1.ElectrodeSeg(filePath=self.directory_labels, patName=self.patName, iLabel=iLabel, numMax=self.numMax, diameterSize=self.diameterSize, spacing=self.spacing, gap=self.gap)
            xxx.pipeline()
            print(xxx.elecPos)
        self.finished.emit()

def savenpy(filePath, patientName):
    dir = f"{filePath}/{patientName}_result"
    # dir1 = f"{filePath}/{patientName}_data"
    elec_dict = {}
    for root, dirs, files in os.walk(dir, topdown=True):
        # print('files:', files)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if 'chnXyzDict.npy' in files:
            files.remove('chnXyzDict.npy')
        for file in files:
            elec_name = file.split('.')[0]
            elec_info = np.loadtxt(os.path.join(root, file))
            
            elec_info = elec_info # [1:, :] # [:,np.array([2,1,0])]
            elec_dict[elec_name] = elec_info
        
    np.save(f"{filePath}/chnXyzDict.npy", elec_dict)

def lookupTable(subdir, patient, ctdir, elec_label):
    annot_dir = f"{subdir}/subjects/{patient}/mri/aparc.a2009s+aseg.mgz"
    lookup_table = f"{subdir}/FreeSurferColorLUT.txt" 
    annot_img = nib.load(annot_dir).get_fdata()
    
    elecs_file = f"{ctdir}/{patient}_result/{elec_label}.txt"
    elecs_xyz = np.loadtxt(elecs_file, dtype='float', comments='#')
    elecs_xyz = elecs_xyz[:, [0, 2, 1]]
    elecs_xyz[:, 0] = 128 - elecs_xyz[:, 0]
    elecs_xyz[:, 1] = 128 - elecs_xyz[:, 1]
    elecs_xyz[:, 2] = 128 + elecs_xyz[:, 2] 
    
    labels = []
    for row in range(elecs_xyz.shape[0]):
        x = elecs_xyz[row, 0]
        y = elecs_xyz[row, 1]
        z = elecs_xyz[row, 2]
        x1 = int(x)
        x2 = math.ceil(x)
        y1 = int(y)
        y2 = math.ceil(y)
        z1 = int(z)
        z2 = math.ceil(z)
        val = [0]
        val.append(annot_img[x1, y1, z1])
        val.append(annot_img[x1, y1, z2])
        val.append(annot_img[x1, y2, z1])
        val.append(annot_img[x1, y2, z2])
        val.append(annot_img[x2, y1, z1])
        val.append(annot_img[x2, y1, z2])
        val.append(annot_img[x2, y2, z1])
        val.append(annot_img[x2, y2, z2])
        val = val[1:]
        labels.append(max(set(val), key = val.count))
    
    # print(labels)
    labels_name = []
    for label in labels:
        with open(lookup_table, 'r') as f:
            lines = f.readlines()
            rows = len(lines)
            for row in range(rows):
                line = lines[row][0: 8]
                b = str(int(label))
                if re.match(b, line):
                    # print(lines[row])
                    a = lines[row][len(b): -16].strip()
                    labels_name.append(a)
                    break
    return labels_name