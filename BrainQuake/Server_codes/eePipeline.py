import os
import nibabel as nib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
from skimage import measure
import scipy
from scipy import ndimage

# patient = 'yejiguo'
# CT_dir = f"/Users/fangcai/Documents/MATLAB/{patient}_test/CT"
# mri_dir = f"/Users/fangcai/Documents/MATLAB/{patient}_test/{patient}/mri"
SUBJECTS_DIR = f"/usr/local/freesurfer/subjects"

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
    print(f"Done!")

def align(
    inp,
    ref,
    xfm=None,
    out=None,
    dof=12,
    searchrad=True,
    bins=256,
    interp=None,
    cost="mutualinfo",
    sch=None,
    wmseg=None,
    init=None,
    finesearch=None,
):
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

def eep(patient):

    ct_dir = f"./data/recv/{patient}"
    # mri_dir = f"/Users/fangcai/Documents/PYTHON/{patient}_test/{patient}/mri"
    mri_dir = f"{SUBJECTS_DIR}/{patient}/mri"

    # mri_convert
    cmd1 = f"mri_convert {mri_dir}/orig.mgz {mri_dir}/orig.nii.gz"
    run(cmd1)

    # Registration
    align(
        inp = f"{ct_dir}/{patient}CT.nii.gz",
        ref = f"{mri_dir}/orig.nii.gz",
        xfm = f"{ct_dir}/fslresults/{patient}invol2refvol.mat",
        out = f"{ct_dir}/fslresults/{patient}outvol.nii.gz",
        dof = 12,
        searchrad = True,
        bins = 256,
        interp = None,
        cost = "normmi",
        sch = None,
        wmseg = None,
        init = None,
        finesearch = None,
    )
    
    # align_nonlinear(
    #     inp = f"{CT_dir}/{patient}CT.nii.gz",
    #     ref = f"{mri_dir}/orig.nii.gz",
    #     xfm = f"{CT_dir}/fslresults/{patient}invol2refvol.mat",
    #     out = f"{CT_dir}/fslresults/{patient}outvolnonlinear.nii.gz",
    #     warp= f"{CT_dir}/fslresults/{patient}invol2refvolnonlinear.mat",
    # )
    
    ## Masking
    CTreg = os.path.join(f"{ct_dir}/fslresults", f"{patient}outvol.nii.gz")
    mask = os.path.join(mri_dir, f"mask.mgz")
    img_ct = nib.load(CTreg)
    img_mask = nib.load(mask)
    data_ct = img_ct.get_fdata()
    data_mask = img_mask.get_fdata()
    # eroding the mask
    data_mask_ero = ndimage.morphology.binary_erosion(data_mask, iterations=10)
    # masking
    data_ct[data_mask_ero==0] = 0
    data_ct[data_ct<0] = 0
    img0 = nib.Nifti1Image(data_ct, img_ct.affine)
    nib.save(img0, os.path.join(f"{ct_dir}/fslresults", f"{patient}intracranial.nii.gz"))
    
    
    
# if __name__ == "__main__":
#     main()