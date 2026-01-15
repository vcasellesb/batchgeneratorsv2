import nibabel as nib
import numpy as np
import torch

from batchgeneratorsv2.transforms.spatial.reg_errors import RegistrationErrorsTransform

transforms = RegistrationErrorsTransform(
    0,
    5 * np.pi/180,
    0,
    0,
    p_deform=1.
)

image_path = '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new_lesions/If_00098.nii.gz'
baseline_path = '/Users/vicentcaselles/work/research/timelessegv2/gen_data_new_lesions/Mb3_00098.nii.gz'
img = nib.load(image_path)
baseline = nib.load(baseline_path)

img_data = np.asanyarray(img.dataobj)
baseline_data = np.asanyarray(baseline.dataobj)

data = torch.stack([
    torch.from_numpy(img_data),
    torch.from_numpy(baseline_data)
])

data_transformed = transforms(**{'image': data})['image']
img_transf = data_transformed[0].numpy()
baseline_transf = data_transformed[1].numpy()

assert np.allclose(img_transf, img_data)

nib.save(nib.Nifti1Image(img_transf, np.eye(4)), 'test_reg_errors_image.nii.gz')
nib.save(nib.Nifti1Image(baseline_transf, np.eye(4)), 'test_reg_errors_baseline.nii.gz')
