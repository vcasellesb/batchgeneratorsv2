from typing import Tuple, List, Union
import torch
from torch.nn import functional as F
import numpy as np

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class RegistrationErrorsTransform(ImageOnlyTransform):

    def __init__(self,
                 reference_channel: int,
                 rotation_bounds: float,
                 translation_fraction: float,
                 scale_fraction: float,
                 resampling_mode: str = 'nearest',
                 padding_mode: str = 'zeros'
                 ):
        """
        :param rotation_bounds: should be in radians
        """

        self.reference_channel = reference_channel

        self.rotation_bounds = (-rotation_bounds, rotation_bounds)
        # Multiply by 2 because grid_sample range is [-1, 1] (size 2)
        # If user wants 10% shift, we need 0.2 in grid coordinates.
        self.translation_fraction = (-translation_fraction * 2, translation_fraction * 2)
        self.scale_fraction = (1 - scale_fraction, 1 + scale_fraction)

        self.resampling_mode = resampling_mode
        self.padding_mode = padding_mode


    def get_parameters(self, **data_dict):
        dim = data_dict['image'].ndim - 1
        angles = [np.random.uniform(*self.rotation_bounds) for _ in range(dim)]
        translations = [np.random.uniform(*self.translation_fraction) for _ in range(dim)]
        scales = [np.random.uniform(*self.scale_fraction) for _ in range(dim)]
        affine = create_affine_matrix_3d(angles, translations, scales)
        return {'affine': affine}

    def _apply_to_image(self, img: torch.Tensor, **params):
        affine = params['affine']
        if affine is None:
            return img

        # torch.affine_grid expects N x 3 x 4 affine matrix
        affine = affine.to(img.device).float().unsqueeze(0)

        n_channels = img.shape[0]
        moving_idx = [i for i in range(n_channels) if i != self.reference_channel]
        if len(moving_idx) == 0:
            return img

        img_to_corrupt = img[moving_idx].unsqueeze(0)

        grid = F.affine_grid(affine, img_to_corrupt.shape, align_corners=False)
        sampled_moving = F.grid_sample(
            img_to_corrupt,
            grid,
            mode=self.resampling_mode,
            padding_mode=self.padding_mode,
            align_corners=False
        )
        img[moving_idx] = sampled_moving.squeeze(0)
        return img


def create_affine_matrix_3d(rotation_angles, translations, scales) -> torch.Tensor:
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])
    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])
    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    RS = Rz @ Ry @ Rx @ np.diag(scales)
    T = np.array(translations)[..., None]
    return torch.cat([torch.from_numpy(RS), torch.from_numpy(T)], dim=1)

if __name__ == "__main__":
    print(
        create_affine_matrix_3d(
            [0.5, 0.5, 0.5],
            [0, 0, 0],
            [1., 1., 1.]
        )
    )