#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import os
import time
import nibabel as nib
import numpy as np
from surfer import Brain
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mayavi import mlab

# filePath = f"/Users/fangcai/Desktop/patients_elecs_reGen/zhangkexuan"
# filePath = f"/Users/fangcai/Documents/PYTHON/{patientName}_test"
filePath1 = '/Applications/freesurfer/7.1.0/subjects'

def vis3D(filePath, patName):
    elecs_xyzDict=np.load(f"{filePath}/{patName}_data/chnXyzDict.npy",allow_pickle=True)[()]
    brain_data=nib.load(f"{filePath1}/{patName}/mri/orig.mgz")
    aff_matrix=brain_data.header.get_affine()
    print(aff_matrix)
    verl,facel=nib.freesurfer.read_geometry(f"{filePath1}/{patName}/surf/lh.pial")
    verr,facer=nib.freesurfer.read_geometry(f"{filePath1}/{patName}/surf/rh.pial")
    
    all_ver=np.concatenate([verl,verr],axis=0)
    tmp_facer=facer+verl.shape[0]
    all_face=np.concatenate([facel,tmp_facer],axis=0)
    vol_center_tmp=np.dot(aff_matrix,np.array([128,128,128,1])[:,None])
    vol_center = vol_center_tmp[:3]
    # print('vol_center_tmp\n', vol_center_tmp)
    # print(vol_center)
    # vol_center=np.dot(aff_matrix,np.array([0,0,0,1])[:,None])[:3]
    reCenter_xyzDict={}
    for ch,xyz in elecs_xyzDict.items():
        # tmp_xyz=np.dot(aff_matrix,np.concatenate([xyz,np.ones((xyz.shape[0],1))],axis=1).T)[:3].T
        # reCenter_xyzDict[ch]=(xyz - vol_center.T) # [:-2, :] # remove the 2 outermost contacts
        reCenter_xyzDict[ch]=xyz
    # # for songzishuo
    # reCenter_xyzDict["E'"]=reCenter_xyzDict["E'"][:-2, :]
    # reCenter_xyzDict["F'"]=reCenter_xyzDict["F'"][:-1, :]
    print('------')
    for k, v in reCenter_xyzDict.items():
        print(k, v.shape)
    
    opacity=0.4
    ambient=0.4225
    specular = 0.3
    specular_power = 20
    diffuse = 0.5
    interpolation='phong'
    mlab.figure(bgcolor=(0.8,0.8,0.8),size=(1500,1500))
    figure=mlab.gcf()
    mesh=mlab.triangular_mesh(all_ver[:,0],all_ver[:,1],all_ver[:,2],all_face,color=(1.,1.,1.),representation='surface',opacity=opacity,line_width=1.)
    # change OpenGL mesh properties for phong point light shading
    mesh.actor.property.ambient = ambient
    mesh.actor.property.specular = specular
    mesh.actor.property.specular_power = specular_power
    mesh.actor.property.diffuse = diffuse
    mesh.actor.property.interpolation = interpolation
    mesh.actor.property.backface_culling = True
    # mesh.scene.light_manager.light_mode = 'vtk'
    if opacity <= 1.0:
        mesh.scene.renderer.trait_set(use_depth_peeling=True)  # , maximum_number_of_peels=100, occlusion_ratio=0.4)
    # Make the mesh look smoother
    for child in mlab.get_engine().scenes[0].children:
        poly_data_normals = child.children[0]
        poly_data_normals.filter.feature_angle = 80.0
    # colormap = ['Greens','Blues','black-white','Purples','blue-red','Greys','pink','summer','winter','jet']
    # colormap = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)][:n_clus]
    # colormap.append((1/255, 1/255, 1/255)) # discarded
    # i=1
    for chnn,xyz in reCenter_xyzDict.items():
        for j in range(xyz.shape[0]):
            mlab.points3d(xyz[j,0], xyz[j,1], xyz[j,2], color=(0,0,0), scale_factor=1.5)
        #mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], colormap=colormap[i], scale_factor=2)
        mlab.text3d(xyz[-1,0]+4,xyz[-1,1]+4,xyz[-1,2]+4,chnn,orient_to_camera=True,color=(0,0,1),line_width=10,scale=2)
        # print(chnn)
        # i = i+1
    # mlab.title("$%s, n_{clus}=%s, n_{discard}=%s$" % (mpn(patientName), n_clus, n_bad), size=0.5)
    mlab.show()

def mayaviView(filePath, surfPath, subname):
    sub_dir = f"{filePath}/{subname}_data"
    recon_dir = f"{surfPath}/{subname}"
    elecs_file_cf = f"{filePath}/{subname}_data/fslresults/chnXyzDict.npy"

    elecs_data_cf = np.load(elecs_file_cf, allow_pickle=True)[()]
    chn_xyz_cf = []
    chn_names_cf = []
    for k, v in elecs_data_cf.items():
        chn_names_cf += [k+str(x+1) for x in range(v.shape[0])]
        chn_xyz_cf.append(v)
    elecs_xyz_cf = np.concatenate(chn_xyz_cf, axis=0)

    orig = nib.load(os.path.join(recon_dir,'mri','orig.mgz'))
    aff_matrix = orig.header.get_affine()
    minus_xyz = np.dot(aff_matrix,np.array([128,128,128,1]).T)[:3].T
    # elecs_xyz_cf = elecs_xyz_cf - minus_xyz

    brain = Brain(subname,'both','pial',subjects_dir=surfPath,offset=False,interaction='terrain',cortex='0.5',alpha=0.05,background='white')
    insula_alpha = 0.05
    brain.add_label('insula',color='C1',hemi='lh',alpha=insula_alpha)
    brain.add_label('insula',color='C1',hemi='rh',alpha=insula_alpha)
    cin_color='green'
    cigulate_alpha=0.05
    brain.add_label('caudalanteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('rostralanteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('isthmuscingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('posteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('caudalanteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('rostralanteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('isthmuscingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('posteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.show_view('lateral')
    mlab.points3d(elecs_xyz_cf[:,0],elecs_xyz_cf[:,1],elecs_xyz_cf[:,2],scale_factor=2,color=(0.4,0.4,0.4))
    mlab.show()

def mayaviViewCheck(filePath, surfPath, subname):
    sub_dir = f"{filePath}/{subname}_data"
    recon_dir = f"{surfPath}/{subname}"
    elecs_file_cf = f"{filePath}/{subname}_data/chnXyzDict.npy"
    elecs_file_wk = f"/Users/fangcai/Desktop/patients_elecs_reGen/{subname}/chnXyzDict.npy"

    elecs_data_cf = np.load(elecs_file_cf, allow_pickle=True)[()]
    chn_xyz_cf = []
    chn_names_cf = []
    for k, v in elecs_data_cf.items():
        chn_names_cf += [k+str(x+1) for x in range(v.shape[0])]
        chn_xyz_cf.append(v)
    elecs_xyz_cf = np.concatenate(chn_xyz_cf, axis=0)

    elecs_data_wk = np.load(elecs_file_wk, allow_pickle=True)[()]
    chn_names_wk = []
    chn_xyz_wk = []
    for k, v in elecs_data_wk.items():
        chn_names_wk += [k+str(x+1) for x in range(v.shape[0])]
        chn_xyz_wk.append(v)
    elecs_xyz_wk = np.concatenate(chn_xyz_wk, axis=0)

    orig = nib.load(os.path.join(recon_dir,'mri','orig.mgz'))
    aff_matrix = orig.header.get_affine()
    minus_xyz = np.dot(aff_matrix,np.array([128,128,128,1]).T)[:3].T
    # elecs_xyz_cf = elecs_xyz_cf - minus_xyz

    orig1 = nib.load(os.path.join(f"/Users/fangcai/Desktop/patients_elecs_reGen/{subname}", 'orig.mgz'))
    aff_matrix1 = orig1.header.get_affine()
    minus_xyz1 = np.dot(aff_matrix1,np.array([128,128,128,1]).T)[:3].T
    elecs_xyz_wk = elecs_xyz_wk - minus_xyz # minus_xyz1 !!!

    brain = Brain(subname,'both','pial',subjects_dir=surfPath,offset=False,interaction='terrain',cortex='0.5',alpha=0.05,background='white')
    insula_alpha = 0.05
    brain.add_label('insula',color='C1',hemi='lh',alpha=insula_alpha)
    brain.add_label('insula',color='C1',hemi='rh',alpha=insula_alpha)
    cin_color='green'
    cigulate_alpha=0.05
    brain.add_label('caudalanteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('rostralanteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('isthmuscingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('posteriorcingulate',color=cin_color,hemi='lh',alpha=cigulate_alpha)
    brain.add_label('caudalanteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('rostralanteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('isthmuscingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    brain.add_label('posteriorcingulate',color=cin_color,hemi='rh',alpha=cigulate_alpha)
    
    brain.show_view('lateral')
    mlab.points3d(elecs_xyz_cf[:,0],elecs_xyz_cf[:,1],elecs_xyz_cf[:,2],scale_factor=2,color=(0.4,0.4,0.4))
    mlab.points3d(elecs_xyz_wk[:,0],elecs_xyz_wk[:,1],elecs_xyz_wk[:,2],scale_factor=2,color=(0.8,0.8,0.8))
    mlab.show()

# if __name__ == '__main__':
#     vis3D(filePath, patName)
