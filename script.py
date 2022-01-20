import numpy as np
from scipy import ndimage as ndi
import slicer
import time
try:
   import nibabel 
except:
  slicer.util.pip_install('nibabel')
  import nibabel 
import numpy as np

try:
   from skimage.measure import label
   from skimage.segmentation import find_boundaries  
except:
    slicer.util.pip_install('scikit-image')
    from skimage.measure import label
    from skimage.segmentation import find_boundaries  

from ThirdParty.utils import nib_to_sitk, sitk_to_nib
import SimpleITK as sitk

def reorient_acquisition(acquisition_order):
    if acquisition_order in ["IS", "SI"]:
        return (2,2)
    elif acquisition_order in ["PA", "AP"]:
        return (1,2)
    elif acquisition_order in ["LR", "RL"]:
        return (0,2)

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def quantisize(images):
    lower = np.percentile(images,0.1)
    upper = np.percentile(images,99.9)
    
    images[images<lower] = lower
    images[images>upper] = upper
    images = (images - lower) / (upper - lower)

    return np.digitize(images, np.arange(0,256)/256).astype('i') 


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1

# Holes structure for 2D and 3D.
_STRUC3 = np.array([[[False, False, False],
        [False,  False, False],
        [False, False, False]],

        [[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]],

        [[False, False, False],
        [False,  False, False],
        [False, False, False]]], dtype=bool)

_STRUC3 = np.swapaxes(_STRUC3, 0, 2)
_STRUC2 = np.ones((1,5,1)).astype(bool)


def levelset_update_2d(u, image):
    c0 = np.mean(image[u==0])
    c1 = np.mean(image[u==1])

    # Image attachment
    du = np.gradient(u)
    abs_du = np.abs(du).sum(0)
    aux = abs_du * ((image - c1)**2 - (image - c0)**2)

    u[aux < 0] = 1
    u[aux > 0] = 0
    
    u = ndi.binary_fill_holes(u).astype(np.int8)
    return u


def levelset_update_3d(u, image, c0=None, c1=None):
    if c0 is None:
        c0 = np.mean(image[u==0])
    if c1 is None:
        c1 = np.mean(image[u==1])

    du = np.gradient(u)
    abs_du = np.abs(du).sum(0)
    aux = abs_du * ((image - c1)**2 - (image - c0)**2)
    u[aux < 0] = 1
    u[aux > 0] = 0
    u = ndi.binary_fill_holes(u, structure=_STRUC3).astype(np.int8)
    u = np.swapaxes(ndi.binary_fill_holes(np.swapaxes(u,0,2),structure=_STRUC3).astype(np.int8),0,2)
    u = np.swapaxes(ndi.binary_fill_holes(np.swapaxes(u,1,2),structure=_STRUC3).astype(np.int8),1,2)
    return u


def morphological_chan_vese_fillhole_2d_new(image: sitk.Image, orientation):
    """Morphological Active Contours without Edges (MorphACWE)
    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images and volumes without well defined
    borders. It is required that the inside of the object looks different on
    average than the outside (i.e., the inner area of the object should be
    darker or lighter than the outer area on average).
    Parameters
    ----------
    image : Input volume (3D SimpleITK image)
            Acquisition orientation (string)
    Returns
    -------
    out : 3D SimpleITK image
        Final segmentation (i.e., the final level set)
    See also
    --------
    circle_level_set, checkerboard_level_set
    Notes
    -----
    This is a modified version of the morphological Chan-Vese algorithm. 
    The algorithm and its theoretical derivation are described in [1]_.
    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """
    img_data_sitk, affine_sitk = sitk_to_nib(image)
    img_nib = nibabel.Nifti1Image(img_data_sitk.squeeze(), affine_sitk)

    img_nib_can = nibabel.as_closest_canonical(img_nib)
    image_data = img_nib_can.get_fdata()
    affine = img_nib_can.affine
    reorient_axis = reorient_acquisition(orientation)
    
    image_data_reoriented = np.swapaxes(image_data, reorient_axis[0], reorient_axis[1])
    image_data_reoriented = quantisize(image_data_reoriented)
    x_shape, y_shape, z_shape = image_data_reoriented.shape
    
    image_data_reoriented_pad = np.zeros([x_shape+2, y_shape+2, z_shape])
    image_data_reoriented_pad[1:-1,1:-1,:] = image_data_reoriented
    u_pad =  np.zeros_like(image_data_reoriented_pad)
    u_pad[1:-1,1:-1,:] = 1

    # 2D Level set
    nb_iterations = max(x_shape,y_shape) // 2

    progressDialog = slicer.util.createProgressDialog(
        value=0,
        maximum=nb_iterations,
        windowTitle='First phase in 2D...',
      )
    start_time = time.time()
    for it in range(nb_iterations):
        u_pad = np.stack([levelset_update_2d(u_pad[...,k], image_data_reoriented_pad[...,k])
                      for k in range(z_shape)], -1)
        progressDialog.setValue(it)
        elapsed_time =  time.time() - start_time
        time_left = nb_iterations * elapsed_time / (it+1) - elapsed_time
        progressDialog.setLabelText(f'Time left: {np.round(time_left,0)} seconds - Nb slices: {z_shape}')
        slicer.app.processEvents()  # necessary?
        if progressDialog.wasCanceled:
            break
    progressDialog.close()
    u = u_pad[1:-1,1:-1,:]

    u[...,u.sum(0).sum(0)>0.99*x_shape*y_shape] = 0
    u = ndi.binary_fill_holes(u, structure=_STRUC3).astype(np.int8)
    u = np.swapaxes(ndi.binary_fill_holes(np.swapaxes(u,0,2),structure=_STRUC3).astype(np.int8),0,2)
    u = np.swapaxes(ndi.binary_fill_holes(np.swapaxes(u,1,2),structure=_STRUC3).astype(np.int8),1,2)
    
    u = ndi.binary_erosion(u).astype(np.int8)
    u = ndi.binary_dilation(u).astype(np.int8)
    
    u = getLargestCC(u).astype(np.int8)
    

     # 3D Level set
    iterations = [25, 10, 5]
    #iterations = [1, 1, 1]
    progressDialog2 = slicer.util.createProgressDialog(
        value=0,
        maximum=sum(iterations),
        windowTitle='Second phase in 3D...',
      )
    progress_bar_i = 0
    start_time = time.time()
    for it in range(3):
        #processed = np.stack([mark_boundaries(u2[...,k], u2[...,k]==1).sum(-1) for k in range(u2.shape[-1])],-1)
        processed = np.stack([find_boundaries(u[...,k],mode='inner') for k in range(u.shape[-1])],-1)
        
        c0 = np.mean(image_data_reoriented[u==0])
        c1 = np.mean(image_data_reoriented[processed>0])
        
        for _ in range(iterations[it]):
            u = levelset_update_3d(u, image_data_reoriented, c0, c1)
            progressDialog2.setValue(progress_bar_i)
            slicer.app.processEvents()
            elapsed_time =  time.time() - start_time
            time_left = sum(iterations) * elapsed_time / (progress_bar_i+1) - elapsed_time
            progressDialog2.setLabelText(f'Time left: {np.round(time_left,0)} seconds')
            progress_bar_i+=1
            if progressDialog2.wasCanceled:
                break

        if progressDialog2.wasCanceled:
            break    
        u = ndi.binary_erosion(u).astype(np.int8)
        u = ndi.binary_dilation(u).astype(np.int8)
        u = getLargestCC(u).astype(np.int8)

    progressDialog2.close()

    u_reorient = np.swapaxes(u, reorient_axis[1], reorient_axis[0])[None,...]
    output = nib_to_sitk(u_reorient, affine)
    return output