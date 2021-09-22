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
    def __init__(self, patch_size: TypePatchSize, num_patches: int, is_shuffled: bool, is_test: bool):
        super().__init__(patch_size)
        self.current_z_slice = 0
        self.remaining_layers = None
        self.num_patches = num_patches
        self.is_shuffled = is_shuffled
        self.is_test = is_test
        self.patches_generated = {}

    def __call__(
            self,
            subject: Subject,
            num_patches: int = None,
            ) -> Generator[Subject, None, None]:
        subject.check_consistent_spatial_shape()

        # if subject['subj_id'] not in self.patches_generated:
        #     self.patches_generated[subject['subj_id']] = 0

        # print('Cropping volume', subject['bvec_volume'])

        if self.num_patches is not None and num_patches is None:
            num_patches = self.num_patches

        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)

        valid_range = subject.spatial_shape[:-1] - self.patch_size[:-1]
        # self.remaining_layers = list(range(subject.spatial_shape[-1]))
        # if self.is_shuffled:
        #     np.random.shuffle(self.remaining_layers)
        # if self.is_test:
        num_patches = subject.spatial_shape[-1]
        patches_left = num_patches if num_patches is not None else True

        entry_key = subject['subj_id'] + "_" + str(subject['bvec_volume'])

        if entry_key in self.patches_generated:
            remaining_layers = [x for x in range(subject.spatial_shape[-1]) if x not in self.patches_generated[entry_key]]
            # remaining_layers = [x for x in range(1) if x not in self.patches_generated[entry_key]]
        else:
            # remaining_layers = [0]
            remaining_layers = list(range(subject.spatial_shape[-1]))

        if not remaining_layers:
            return

        while patches_left:
            index_ini = [torch.randint(x + 1, (1,)).item() for x in valid_range]
            
            if self.is_shuffled:
                np.random.shuffle(remaining_layers)

            current_z_slice = remaining_layers.pop()
            
            if entry_key not in self.patches_generated:
                self.patches_generated[entry_key] = [current_z_slice]
            else:
                self.patches_generated[entry_key].append(current_z_slice)


            index_ini = [*index_ini, current_z_slice]
            
            index_ini_array = np.asarray(index_ini)
            if patches_left:
                patches_left -= 1
            p = self.extract_patch(subject, index_ini_array)
            # self.patches_generated[subject['subj_id']] += 1
            
            yield p
