import typing as ty
import torch
from numbers import Number

from ..base.basic_transform import SegOnlyTransform


class DummyError(Exception):
    pass


class DropLabels(SegOnlyTransform):

    def __init__(self,
                 labels: ty.Sequence[int],
                 classes_to_keep: int | ty.Sequence[int] | None,
                 classes_to_remove: int | ty.Sequence[int] | None
                 ):

        super().__init__()

        if (
            classes_to_keep is None and classes_to_remove is None
        ):
            raise DummyError

        maxlabel = max(labels) + 1
        if classes_to_keep is not None:
            lut = torch.zeros(maxlabel, dtype=torch.long)
            classes_to_keep = torch.as_tensor([classes_to_keep] if isinstance(classes_to_keep, Number) else classes_to_keep).long()
            lut[classes_to_keep] = classes_to_keep
        else:
            lut = torch.arange(maxlabel, dtype=torch.long)
            classes_to_remove = torch.as_tensor([classes_to_remove] if isinstance(classes_to_remove, Number) else classes_to_remove).long()
            lut[classes_to_remove] = 0

        self.lut = lut


    def get_parameters(self, **data_dict):
        return {'lut': self.lut.clone().to(data_dict['segmentation'].device)}

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params):
        lut = params.get('lut')
        return lut[segmentation.long()].to(segmentation.dtype)

if __name__ == "__main__":
    x = torch.randint(0, 5, size=(1, 128, 128, 128))
    labels = list(range(5))
    classes_to_keep = [1]
    t = DropLabels(labels, classes_to_keep, None)
    y1 = t(segmentation=x)['segmentation']


    t = DropLabels(labels, None, [i for i in labels if i not in classes_to_keep])
    y2 = t(segmentation=x)['segmentation']

    x[torch.isin(x, torch.tensor(classes_to_keep), invert=True)] = 0

    assert torch.equal(y1, y2)
    assert torch.equal(y1, x)
    
