import sys
import nibabel as nib
import torch
import numpy as np

from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

lesion_mask = sys.argv[1]
lesion_mask: nib.Nifti1Image = nib.load(lesion_mask)
affine = lesion_mask.affine

patch_size = lesion_mask.header.get_data_shape()
lesion_mask = (np.asanyarray(lesion_mask.dataobj) == 7).astype(np.int8)

t = SpatialTransform(patch_size,
                     0,
                     random_crop=False,
                     p_elastic_deform=1,
                     elastic_deform_scale=(0.2, 1),
                     elastic_deform_magnitude=(0.2, 1),
                     p_rotation=0,
                     p_scaling=0,
                     bg_style_seg_sampling=False)

transformed_lesion_mask = t(**{
    'image': torch.from_numpy(lesion_mask[None]).float(),
    'segmentation': torch.from_numpy(lesion_mask[None])
})['segmentation']

transformed_lesion_mask = transformed_lesion_mask.numpy()[0]

print((transformed_lesion_mask == lesion_mask).all())

nib.save(
    nib.Nifti1Image(transformed_lesion_mask, affine=affine),
    sys.argv[1].replace('.nii.gz', '_def.nii.gz')
)