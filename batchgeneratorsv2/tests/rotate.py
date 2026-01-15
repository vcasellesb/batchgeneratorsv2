import numpy as np
import torch
from torch.nn import functional as F
import nibabel as nib

from batchgeneratorsv2.transforms.spatial.reg_errors import create_affine_matrix_3d

def rotate_data(data: np.ndarray, rotation_angles_in_degrees):

    assert data.ndim == 3 and len(rotation_angles_in_degrees) == data.ndim
    rotation_angles_in_radians = [r * np.pi / 180 for r in rotation_angles_in_degrees]
    affine_matrix = create_affine_matrix_3d(rotation_angles_in_radians,
                                               translations=[0] * data.ndim,
                                               scales=[1] * data.ndim).unsqueeze(0)

    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(affine_matrix, size=data.shape, align_corners=False).float()


    data_rotated = F.grid_sample(
        data.float(),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    return data_rotated, affine_matrix

def get_updated_affine(old_affine, pytorch_A, shape):
    """
    old_affine: 4x4 numpy array (NIfTI header)
    pytorch_A: 3x4 or 4x4 matrix used in affine_grid (Source-to-Target mapping)
    shape: (D, H, W) or (H, W) - spatial dimensions of the image
    """
    # 1. Construct the Normalization Matrix N (Voxel -> [-1, 1])
    # Formula: norm = 2 * (voxel / (size - 1)) - 1
    # This is a Scale and a Shift.
    
    D, H, W = shape
    
    # Scale factors: 2 / (dim - 1)
    sx = 2 / (W - 1)
    sy = 2 / (H - 1)
    sz = 2 / (D - 1)
    
    # N matrix (Voxel to Normalized)
    # Note: PyTorch uses (x, y, z) order, which corresponds to (W, H, D)
    N = np.eye(4)
    N[0, 0] = sx; N[0, 3] = -1
    N[1, 1] = sy; N[1, 3] = -1
    N[2, 2] = sz; N[2, 3] = -1

    # N inverse (Normalized to Voxel)
    N_inv = np.linalg.inv(N)

    # 2. Ensure pytorch_A is a 4x4 numpy array
    if torch.is_tensor(pytorch_A):
        pytorch_A = pytorch_A.detach().cpu().numpy()
    
    # If A is 3x4 (standard PyTorch output), add the [0,0,0,1] row
    if pytorch_A.shape == (3, 4):
        A_4x4 = np.eye(4)
        A_4x4[:3, :] = pytorch_A
    else:
        A_4x4 = pytorch_A

    # 3. Create the Voxel-Space Transform M
    # Sandwich: Normalized -> Voxel (Input) -> Apply A -> Normalized -> Voxel (Output) 
    # WAIT! affine_grid maps Target -> Source.
    # So V_source = M @ V_target
    # We want M that corresponds to A.
    # A maps Norm_target -> Norm_source
    # So M = N_inv @ A @ N

    M = N_inv @ A_4x4 @ N

    # 4. Update the Affine
    # World = T_old @ V_source
    # World = T_old @ (M @ V_target)
    # T_new = T_old @ M
    
    new_affine = old_affine @ M
    
    return new_affine

def apply_to_image(data, affine, rotation_angles):
    result_torch, affine_matrix = rotate_data(data, rotation_angles)

    result_numpy = result_torch.squeeze().squeeze().detach().cpu().numpy()
    new_affine = get_updated_affine(affine, affine_matrix.squeeze(), result_numpy.shape)
    return result_numpy, new_affine


def main(image_path):
    img = nib.load(image_path)
    affine = img.affine
    data_original: np.ndarray = np.asanyarray(img.dataobj)

    rotation_angles = [5, 0, 0]
    result_numpy, new_affine = apply_to_image(data_original.copy(), affine, rotation_angles)
    nib.save(
        nib.Nifti1Image(result_numpy, new_affine),
        f'test_rotation_45_0_0.nii.gz'
    )

    rotation_angles = [5, 5, 0]
    result_numpy, new_affine = apply_to_image(data_original.copy(), affine, rotation_angles)

    nib.save(
        nib.Nifti1Image(result_numpy, new_affine),
        f'test_rotation_45_45_0].nii.gz'
    )

    rotation_angles = [0, 5, 5]
    result_numpy, new_affine = apply_to_image(data_original.copy(), affine, rotation_angles)
    nib.save(
        nib.Nifti1Image(result_numpy, new_affine),
        f'test_rotation_0_45_45.nii.gz'
    )


if __name__ == "__main__":
    main('/Users/vicentcaselles/work/research/project_MARCOS/Multiple-Sclerosis-TIMILS/subj9/flair_bfc.nii.gz')