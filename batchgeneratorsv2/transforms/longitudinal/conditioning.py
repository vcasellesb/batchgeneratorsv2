import typing as ty
import torch
import numpy as np

from ..base.basic_transform import BasicTransform
from .ciao import CiaoBaseline


def dummy_flm(**data_dict):
    return data_dict

class ConditioningTransform(BasicTransform):

    def __init__(self,
                 p_cross_sectional: float,
                 p_healthy_prior: float,
                 flm_already_computed: bool,
                 target_class: int = None,
                 regions_where_class_cannot_be: ty.Sequence[int] = None
                 ):

        super().__init__()
        assert p_cross_sectional >= 0 and p_healthy_prior >= 0 and p_cross_sectional + p_healthy_prior <= 1.
        self.weights = [p_cross_sectional, p_healthy_prior, (1 - p_cross_sectional - p_healthy_prior)]
        self.meanings = ['cross_sectional', 'healthy_prior', 'fake_lesion_mask']

        self.cross_sectional_transform = CiaoBaseline(p_per_channel=[0., 1.])
        if flm_already_computed:
            self.flm_transform = dummy_flm
        else:
            assert target_class is not None and regions_where_class_cannot_be is not None
            from .flm import FakeLesionMask

            self.flm_transform = FakeLesionMask(target_class=target_class, regions_where_class_cannot_be=regions_where_class_cannot_be)

    def get_parameters(self, **data_dict):
        params = {'do_cross_sectional': False, 'do_healthy_prior': False, 'do_fake_lesion_mask': False}
        choice = np.random.choice(len(self.weights), p=self.weights)
        params[f'do_{self.meanings[choice]}'] = True
        return params

    def _apply_to_image(self, img, segmentation, **params):
        # importantly, this assumes that each training case, where this is run, has a baseline mask
        if params['do_fake_lesion_mask']:
            img = self.flm_transform(image=img, segmentation=segmentation)['image']
            conditioning_factor = torch.tensor([True])
        else:
            img = self.cross_sectional_transform(image=img)['image']
            conditioning_factor = torch.tensor([not params['do_cross_sectional']])

        conditioning_factor = conditioning_factor.to(img.device)

        return img, conditioning_factor

    def apply(self, data_dict, **params):
        img, conditioning_factor = self._apply_to_image(data_dict['image'], data_dict.get('segmentation'), **params)
        data_dict['image'] = img
        data_dict['conditioning'] = conditioning_factor
        return data_dict


if __name__ == "__main__":
    img = torch.empty((2, 128, 124, 226))
    img[0].normal_()
    img[1].random_(0, 2)

    c = ConditioningTransform(1., 0., True, None, None)
    got = c(image=img.clone())
    assert torch.equal(got['image'][1], torch.zeros_like(img[1])) and got['conditioning'].item() == 0.

    c = ConditioningTransform(0., 1., True, None, None)
    got = c(image=img.clone())
    assert torch.equal(got['image'][1], torch.zeros_like(img[1])) and got['conditioning'].item() == 1.

    c = ConditioningTransform(0., 0., True, None, None)
    got = c(image=img.clone())
    assert torch.equal(got['image'][1], img[1]) and got['conditioning'].item() == 1.
    
