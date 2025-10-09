from typing import Iterable

import torch

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class CiaoBaseline(ImageOnlyTransform):
    """
    Removes baseline -- this is thought for trying to make a network learn to work in both
    longitudinal and cross-sectional settings.
    """
    def __init__(
        self,
        p_per_channel: float | Iterable[float]
    ):
        self.p_per_channel = torch.tensor(p_per_channel, dtype=float)


    def get_parameters(self, **data_dict):
        shape = data_dict['image'].shape
        apply_per_channel = torch.rand(shape[0]) < self.p_per_channel
        return {'apply_per_channel': apply_per_channel}


    def _apply_to_image(self, img: torch.Tensor, **params):
        img[params.get('apply_per_channel')] *= 0
        return img


if __name__ == "__main__":
    x = torch.randn((2, 128, 128, 128))
    ciao = CiaoBaseline([0, 1])
    out = ciao(image=x)

    assert x[1].sum() == 0. and x[1].max() == 0.
    assert x[0].max() != 0 and x[0].min() != 0
