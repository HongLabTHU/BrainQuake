#! /usr/bin/python3.7
# -- coding: utf-8 -- **

### Here are a set of functions used in elec_pipe
### and a set of qthread class for elec_main_gui

import sys
import os
import re
import math
import numpy as np
from numpy import ndarray
import nibabel as nib
from scipy import ndimage
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import LinearRegression, Lasso
from PyQt5.QtCore import QThread, pyqtSignal
# import matplotlib
# matplotlib.use("Qt5Agg")
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D, art3d
# import electrode

CMD_Hough3D = './hough-3d-lines/hough3dlines'

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
        xs, ys, zs = dataExtraction(intraFile=intra_file, thre=self.thre)
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
            # xxx = electrode.ElectrodeSeg(filePath=self.directory_labels, patName=self.patName, iLabel=iLabel, numMax=self.numMax, diameterSize=self.diameterSize, spacing=self.spacing, gap=self.gap)
            xxx = ElectrodeSeg(filePath=self.directory_labels, patName=self.patName, iLabel=iLabel, numMax=self.numMax, diameterSize=self.diameterSize, spacing=self.spacing, gap=self.gap)
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


class ElectrodeSeg:
    def __init__(self, filePath, patName, iLabel, numMax, diameterSize, spacing, gap):
        super(ElectrodeSeg, self).__init__()
        # set up input initials
        self.filePath = filePath
        self.patientName = patName
        
        raw_flag = 0 # check for the filepath existance
        for root, dirs, files in os.walk(self.filePath):
            for filename in files:
                if re.search(r'CT_intra.nii.gz', filename):
                    raw_flag = 1
                    self.rawDataPath = f"{self.filePath}/{filename}"
                    break
        if not raw_flag:
            sys.exit()

        label_flag = 0
        for root, dirs, files in os.walk(self.filePath):
            for filename in files:
                if re.search(r'_labels.npy', filename):
                    label_flag = 1
                    self.labelsPath = f"{self.filePath}/{filename}"
                    break
        if not label_flag:
            sys.exit()
        
        self.rawData = nib.load(self.rawDataPath).get_fdata()
        self.labels = np.load(self.labelsPath)
        self.iLabel = iLabel
        self.numMax = numMax
        self.diameterSize = diameterSize
        self.spacing = spacing
        self.gap = gap

        # some calculations to get the rest initials
        self.labelValues = np.unique(self.labels)
        self.numElecs = len(self.labelValues) - 1
        if self.numElecs > 8: # remove 'I' from the alphabet list, a trivial custom not to name the electrode 'I'
            self.alphaList = [chr(i) for i in range(65, 66+self.numElecs)]
            self.alphaList.pop(8)
        else:
            self.alphaList = [chr(i) for i in range(65, 65+self.numElecs)]
        self.iValue = self.labelValues[self.iLabel]
        self.nameLabel = self.alphaList[self.iLabel-1]
        data_elec = np.copy(self.labels)
        data_elec[np.where(self.labels != self.iValue)] = 0 ## isolate a single cluster of voxels belonging to the ith electrode
        self.xs, self.ys, self.zs = np.where(data_elec != 0) 
        self.pos_elec = np.transpose(np.vstack((self.xs, self.ys, self.zs))) ## positions of these voxels
        ### test!
        data_elec1 = np.copy(self.labels)
        data_elec1[np.where(self.labels == self.iValue)] = 0
        self.xrest, self.yrest, self.zrest = np.where(data_elec1 != 0)
        self.rawData[self.xrest, self.yrest, self.zrest] = 0
        ### test!
        self.rawData_single = self.rawData
        xmin = np.amin(self.xs)
        xmax = np.amax(self.xs)
        ymin = np.amin(self.ys)
        ymax = np.amax(self.ys)
        zmin = np.amin(self.zs)
        zmax = np.amax(self.zs)
        # self.rawData_single[self.xs, self.ys, self.zs] = self.rawData_single[self.xs, self.ys, self.zs] * 3
        self.rawData_single[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = self.rawData_single[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] * 3

        self.resultPath = f"{self.filePath}/{self.patientName}_result"
        if not os.path.exists(self.resultPath):
            os.mkdir(self.resultPath)
        self.resultFile = f"{self.resultPath}/{self.nameLabel}.txt"
        self.elecPos = [0, 0, 0]
        self.headStart = [0, 0, 0]
        self.targetPoint = [0, 0, 0]
        self.regressInfo = [0, 0, 0, 0]
    
    def pipeline(self):
        self.startPoint()
        self.contactPoint(1)
        self.regression()
        for j in np.arange(self.numMax - 1):
            # if self.rawData[int(round(self.elecPos[-1,0])), int(round(self.elecPos[-1,1])), int(round(self.elecPos[-1,2]))] == 0:
            #     self.elecPos = self.elecPos[0:-1, :]
            #     break
            if int(self.elecPos[-1,0])==int(self.elecPos[-2,0]) and int(self.elecPos[-1,1])==int(self.elecPos[-2,1]) and int(self.elecPos[-1,2])==int(self.elecPos[-2,2]):
                self.elecPos = self.elecPos[0:-1, :]
                break
            self.step()
            if self.flag_step_stop:
                break
        self.elecPos = self.elecPos[1:, :]
        # print(self.elecPos)
        self.resulting()
        # return self.elecPos
    
    def resulting(self):
        self.elecPos_true = np.copy(self.elecPos)
        self.elecPos_true[:, 0] = 128 - self.elecPos[:, 0]
        self.elecPos_true[:, 1] = 128 - self.elecPos[:, 1]
        self.elecPos_true[:, 2] = self.elecPos[:, 2] - 128
        self.elecPos_true = self.elecPos_true[:, [0, 2, 1]]
        
        self.elecFilepath = os.path.join(self.filePath, f"{self.patientName}_result")
        if not os.path.exists(self.elecFilepath):
            os.mkdir(self.elecFilepath)
        else:
            self.elecFile = os.path.join(self.elecFilepath, f"{self.nameLabel}.txt")
            with open(self.elecFile, "ab") as f:
                f.seek(0)
                f.truncate()
                # f.write(b"\n")
                np.savetxt(f, self.elecPos_true, fmt='%10.8f', delimiter=' ', newline='\n', header=f"{self.elecPos_true.shape[0]}")


    ## target point functions
    def startPoint(self):
        ## firstly find a voxel near the target
        x = [np.max(self.xs), np.min(self.xs)]
        y = [np.max(self.ys), np.min(self.ys)]
        z = [np.max(self.zs), np.min(self.zs)]
        self.reg1 = LinearRegression().fit(X=self.xs.reshape(-1,1), y=self.ys) # x-y
        self.reg2 = LinearRegression().fit(X=self.xs.reshape(-1,1), y=self.zs) # x-z
        self.reg3 = LinearRegression().fit(X=self.ys.reshape(-1,1), y=self.zs) # y-z

        coefs = [abs(self.reg1.coef_), abs(self.reg2.coef_), abs(self.reg3.coef_)]
        coef_min = coefs.index(min(coefs))
        if coef_min == 0:
            index = [0 if self.reg2.coef_>0 else 1, 0 if self.reg3.coef_>0 else 1, 0]
        elif coef_min == 1:
            index = [0 if self.reg1.coef_>0 else 1, 0, 0 if self.reg3.coef_>0 else 1]
        else:
            index = [0, 0 if self.reg1.coef_>0 else 1, 0 if self.reg2.coef_>0 else 1]
        indexreverse = [~index[0], ~index[1], ~index[2]]

        point1 = np.array([x[index[0]], y[index[1]], z[index[2]]])
        point2 = np.array([x[indexreverse[0]], y[indexreverse[1]], z[indexreverse[2]]])
        center = 127.5 * np.ones(3)
        diff1 = point1 - center
        diff2 = point2 - center
        headStart = point2 if np.sum(np.transpose(diff1)*diff1) > np.sum(np.transpose(diff2)*diff2) else point1
        self.direction = indexreverse if np.sum(np.transpose(diff1)*diff1) > np.sum(np.transpose(diff2)*diff2) else index

        ## secondly specify a target voxel in label voxels
        diffs = self.pos_elec - headStart
        diffs2 = np.power(diffs[:,0], 2) + np.power(diffs[:,1], 2) + np.power(diffs[:,2], 2)
        headPointPos = np.argmin(diffs2)
        self.headStart = self.pos_elec[headPointPos, :]
    
    def converge(self, x, y, z):
        ## converge to the mass center of a cluster of voxels
        n = self.diameterSize
        delta = math.ceil(round((n - 1) / 2, 1)) # represent the radius of the electrode contact
        ## extract a cubic ROI of the raw CT data
        seq_s = np.arange(x - delta, x + delta + 1)
        seq_r = np.arange(y - delta, y + delta + 1)
        seq_c = np.arange(z - delta, z + delta + 1)

        if not ((np.array(seq_s) > 0).all() and (np.array(seq_r) > 0).all() and (np.array(seq_c) > 0).all()):
            print('Error: index too small 0!')
            return 0, 0, 0
        elif not ((np.array(seq_s) < 256).all() and (np.array(seq_r) < 256).all() and (np.array(seq_c) < 256).all()):
            print('Error: index too large 256!')
            return 0, 0, 0
        
        else:
            ## extract the ROI cubic
            # test!!!
            matrixVoxels = self.rawData_local[seq_s[0]:seq_s[-1]+1, seq_r[0]:seq_r[-1]+1, seq_c[0]:seq_c[-1]+1]
            sumVoxels = np.sum(matrixVoxels)

            if (np.sum(matrixVoxels)== 0):
                print('Error: Converge to non-elec region!')
                return 0, 0, 0
            else:
                f = np.zeros((1, 4))
                for index, element in np.ndenumerate(matrixVoxels):
                    x, y, z = index
                    tmp = np.array([x+seq_s[0], y+seq_r[0], z+seq_c[0], element])
                    f = np.vstack((f, tmp))
                f = f[1:]
                CM = np.average(f[:,:3], axis=0, weights=f[:,3])
                C100 = CM[0]
                C010 = CM[1]
                C001 = CM[2]
                
                x1 = C100
                y1 = C010
                z1 = C001
                return x1, y1, z1
    
    def contactPoint(self, target):
        ## converge to an electrode contact position
        x0 = self.headStart[0] if target == 1 else self.x0
        y0 = self.headStart[1] if target == 1 else self.y0
        z0 = self.headStart[2] if target == 1 else self.z0
        
        x = int(round(x0))
        y = int(round(y0))
        z = int(round(z0))
        print(f"initial start voxel:({x0}, {y0}, {z0})")
        
        # test!!!
        self.rawData_local = self.rawData_single
        diff_array = self.pos_elec - np.array([x0, y0, z0])
        elec_diffs = np.sqrt(np.dot(diff_array, np.transpose(diff_array)).diagonal())
        ind_diffs = np.where(elec_diffs <= 2)
        self.rawData_local[self.xs[ind_diffs], self.ys[ind_diffs], self.zs[ind_diffs]] = self.rawData_local[self.xs[ind_diffs], self.ys[ind_diffs], self.zs[ind_diffs]] * 2
        (x1, y1, z1) = self.converge(x, y, z)
        itr = 1
        flag_convergence = 0
        while not ((x==int(round(x1))) and (y==int(round(y1))) and (z==int(round(z1)))):
            x = int(round(x1))
            y = int(round(y1))
            z = int(round(z1))
            (x1, y1, z1) = self.converge(x, y, z)
            itr = itr + 1
            if itr > 5:
                flag_convergence = 1
                break
        
        print(f"Convergent center voxel coordinates:({x1},{y1},{z1})")
        print(f"Convergent center voxel value:{self.rawData[int(round(x1)), int(round(y1)), int(round(z1))]}")
        
        self.flag_step_stop = 0
        if (x1, y1, z1) == (0, 0, 0):
            self.flag_step_stop = 1
            print('here1,converged to 0!')
            # self.elecPos = np.vstack([self.elecPos, [x1, y1, z1]])
        
        else:
            if not flag_convergence:
                print('here2,converged normally!') 
                self.targetPoint = [x1, y1, z1] if target == 1 else self.targetPoint
                self.elecPos = np.vstack([self.elecPos, [x1, y1, z1]])
            else:
                print('here3, maybe not convergent!') 
                self.targetPoint = [x1, y1, z1] if target == 1 else self.targetPoint
                self.elecPos = np.vstack([self.elecPos, [x1, y1, z1]])

    def regression(self):
        ## regress an electrode and find the axis direction
        X = np.transpose(np.vstack((self.xs, self.ys)))
        y = self.zs
        
        forcedX = np.transpose(np.array([self.targetPoint[0], self.targetPoint[1]]))
        forcedy = self.targetPoint[2]
        
        ## implant a contraint regression, forcing on the head point
        X = X - forcedX
        y = y - forcedy
        reg = Lasso(fit_intercept=False).fit(X=X, y=y)
        reg.intercept_ = reg.intercept_ + forcedy - np.dot(forcedX, reg.coef_)
        ## regression between x and y
        reg2 = LinearRegression(fit_intercept=True).fit(X=self.xs.reshape(-1,1), y=self.ys)
        
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.coef2 = reg2.coef_
        self.intercept2 = reg2.intercept_
    
    def step(self):
        ## step out along the electrode axis
        dis = self.spacing # initial step size

        # delta_x = np.sqrt(np.power(dis, 2) / (1 + np.power(self.coef2[0],2) + np.power(np.dot(self.coef, np.array([1, self.coef2[0]])) ,2)))
        # delta_y = np.dot(self.coef2[0], delta_x)
        # delta_z = np.dot(self.coef, np.array([1, self.coef2[0]])) * delta_x
        
        diff_x = np.max(self.xs) - np.min(self.xs)
        diff_y = np.max(self.ys) - np.min(self.ys)
        diff_z = np.max(self.zs) - np.min(self.zs)
        a = np.power(diff_x,2) + np.power(diff_y,2) + np.power(diff_z,2)
        delta_x = diff_x * np.sqrt(np.power(dis,2) / a)
        delta_y = diff_y * np.sqrt(np.power(dis,2) / a)
        delta_z = diff_z * np.sqrt(np.power(dis,2) / a)

        # delta_x = self.reg2.coef_ * np.sqrt(np.power(dis,2) / (1 + np.power(self.reg2.coef_,2) + np.power(self.reg3.coef_,2)))
        # delta_y = self.reg3.coef_ * np.sqrt(np.power(dis,2) / (1 + np.power(self.reg2.coef_,2) + np.power(self.reg3.coef_,2)))
        # delta_z = np.sqrt(np.power(dis,2) / (1 + np.power(self.reg2.coef_,2) + np.power(self.reg3.coef_,2)))

        self.x0 = np.int(self.elecPos[-1,0] - np.round(delta_x)) if ((self.direction[0]==-2) or (self.direction[0]==0)) else np.int(self.elecPos[-1,0] + np.round(delta_x))
        self.y0 = np.int(self.elecPos[-1,1] - np.round(delta_y)) if ((self.direction[1]==-2) or (self.direction[1]==0)) else np.int(self.elecPos[-1,1] + np.round(delta_y))
        self.z0 = np.int(self.elecPos[-1,2] - np.round(delta_z)) if ((self.direction[2]==-2) or (self.direction[2]==0)) else np.int(self.elecPos[-1,2] + np.round(delta_z))
        
        self.contactPoint(0)
