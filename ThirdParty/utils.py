##### CODE FROM TorchIO - Original Author: Fernando Pérez-García
# https://github.com/fepegar/torchio/blob/1bbf99e90cd06112c092a1fc227dedd5deb256ba/torchio/data/io.py




import numpy as np
import SimpleITK as sitk
import nibabel 
from typing import Tuple, Union, Optional

# Matrices used to switch between LPS and RAS
FLIPXY_33 = np.diag([-1, -1, 1])
FLIPXY_44 = np.diag([-1, -1, 1, 1])
TypeData = np.ndarray
TypeTripletFloat = Tuple[float, float, float]
TypeDirection2D = Tuple[float, float, float, float]
TypeDirection3D = Tuple[
    float, float, float, float, float, float, float, float, float]
TypeDirection = Union[TypeDirection2D, TypeDirection3D]

def get_rotation_and_spacing_from_affine(
        affine: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def nib_to_sitk(
        data: TypeData,
        affine: TypeData,
        force_3d: bool = False,
        force_4d: bool = False,
        ) -> sitk.Image:
    """Create a SimpleITK image from a tensor and a 4x4 affine matrix."""
    if data.ndim != 4:
        shape = tuple(data.shape)
        raise ValueError(f'Input must be 4D, but has shape {shape}')
    # Possibilities
    # (1, w, h, 1)
    # (c, w, h, 1)
    # (1, w, h, 1)
    # (c, w, h, d)
    array = np.asarray(data)
    affine = np.asarray(affine).astype(np.float64)

    is_multichannel = array.shape[0] > 1 and not force_4d
    is_2d = array.shape[3] == 1 and not force_3d
    if is_2d:
        array = array[..., 0]
    if not is_multichannel and not force_4d:
        array = array[0]
    array = array.transpose()  # (W, H, D, C) or (W, H, D)
    image = sitk.GetImageFromArray(array, isVector=is_multichannel)

    origin, spacing, direction = get_sitk_metadata_from_ras_affine(
        affine,
        is_2d=is_2d,
    )
    image.SetOrigin(origin)  # should I add a 4th value if force_4d?
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    if data.ndim == 4:
        assert image.GetNumberOfComponentsPerPixel() == data.shape[0]
    num_spatial_dims = 2 if is_2d else 3
    assert image.GetSize() == data.shape[1:1 + num_spatial_dims]

    return image


def sitk_to_nib(
        image: sitk.Image,
        keepdim: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    data = sitk.GetArrayFromImage(image).transpose()
    data = check_uint_to_int(data)
    num_components = image.GetNumberOfComponentsPerPixel()
    if num_components == 1:
        data = data[np.newaxis]  # add channels dimension
    input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 2:
        data = data[..., np.newaxis]
    elif input_spatial_dims == 4:  # probably a bad NIfTI (1, sx, sy, sz, c)
        # Try to fix it
        num_components = data.shape[-1]
        data = data[0]
        data = data.transpose(3, 0, 1, 2)
        input_spatial_dims = 3
    assert data.shape[0] == num_components
    affine = get_ras_affine_from_sitk(image)
    return data, affine


def get_ras_affine_from_sitk(
        sitk_object: Union[sitk.Image, sitk.ImageFileReader],
        ) -> np.ndarray:
    spacing = np.array(sitk_object.GetSpacing())
    direction_lps = np.array(sitk_object.GetDirection())
    origin_lps = np.array(sitk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    rotation_ras = np.dot(FLIPXY_33, rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(FLIPXY_33, origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine



def get_sitk_metadata_from_ras_affine(
        affine: np.ndarray,
        is_2d: bool = False,
        lps: bool = True,
        ) -> Tuple[TypeTripletFloat, TypeTripletFloat, TypeDirection]:
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_lps = np.dot(FLIPXY_33, origin_ras)
    direction_lps = np.dot(FLIPXY_33, direction_ras)
    if is_2d:  # ignore orientation if 2D (1, W, H, 1)
        direction_lps = np.diag((-1, -1)).astype(np.float64)
        direction_ras = np.diag((1, 1)).astype(np.float64)
    origin_array = origin_lps if lps else origin_ras
    direction_array = direction_lps if lps else direction_ras
    direction_array = direction_array.flatten()
    # The following are to comply with typing hints
    # (there must be prettier ways to do this)
    ox, oy, oz = origin_array
    sx, sy, sz = spacing_array
    if is_2d:
        d1, d2, d3, d4 = direction_array
        direction = d1, d2, d3, d4
    else:
        d1, d2, d3, d4, d5, d6, d7, d8, d9 = direction_array
        direction = d1, d2, d3, d4, d5, d6, d7, d8, d9
    origin = ox, oy, oz
    spacing = sx, sy, sz
    return origin, spacing, direction


def check_uint_to_int(array):
    # This is because PyTorch won't take uint16 nor uint32
    if array.dtype == np.uint16:
        return array.astype(np.int32)
    if array.dtype == np.uint32:
        return array.astype(np.int64)
    return array