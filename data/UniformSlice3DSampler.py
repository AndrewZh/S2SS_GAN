import torch
from torchio import Subject
from torchio.data.sampler import UniformSampler
from torchio.typing import TypePatchSize
from torchio.utils import to_tuple
from typing import Generator, Sequence

import numpy as np

class UniformSlice3DSampler(UniformSampler):
    """Randomly extract patches from a volume with uniform probability.
    Args:
        patch_size: See :py:class:`~torchio.data.PatchSampler`.
    """
    def __init__(self, patch_size: TypePatchSize, num_patches: int):
        super().__init__(patch_size)
        self.current_z_slice = 0
        self.remaining_layers = None
        self.num_patches = num_patches

    def __call__(
            self,
            subject: Subject,
            num_patches: int = None,
            ) -> Generator[Subject, None, None]:
        subject.check_consistent_spatial_shape()

        if self.num_patches is not None and num_patches is None:
            num_patches = self.num_patches

        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)

        valid_range = subject.spatial_shape[:-1] - self.patch_size[:-1]
        self.remaining_layers = list(range(subject.spatial_shape[-1]))
        np.random.shuffle(self.remaining_layers)
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            index_ini = [
                torch.randint(x + 1, (1,)).item()
                for x in valid_range
            ]
            if not self.remaining_layers:
                self.remaining_layers = list(range(subject.spatial_shape[-1]))
                np.random.shuffle(self.remaining_layers)
            current_z_slice = self.remaining_layers.pop()
            index_ini = [*index_ini, current_z_slice]
            
            index_ini_array = np.asarray(index_ini)
            if patches_left:
                patches_left -= 1
            p = self.extract_patch(subject, index_ini_array)
            yield p
