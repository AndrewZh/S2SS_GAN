from torchio import ScalarImage, Subject
from typing import Generator
import os.path as osp
import os
import dipy.io as dio
import numpy as np
import nibabel as nib
import torch


# TODO: pick single b0 volume
def __get_bvolumes_bvecs(bval, all_bvals, dwi, bvecs):
    b_mask = all_bvals == bval
    b_volumes = np.moveaxis(dwi[:,:,:,b_mask],-1,0)
    b_vecs = bvecs[b_mask,:]
    
    if bval == 0:
        b_volumes = np.expand_dims(b_volumes[0,...], axis=0)
        b_vecs = b_vecs[0,...]

    return b_volumes, b_vecs

def subject_generator(data_dir, crop_size) -> Generator[Subject, None, None]:
    subj_ids = os.listdir(data_dir)

    for subj_id in subj_ids[:1]: # sina: why iteration over subjects here?
        # if subj_id== '715041': # sina: just take one subject 
        subj_dir = osp.join(data_dir, subj_id)

        bvals, bvecs = dio.read_bvals_bvecs(osp.join(subj_dir, 'bvals'),
                                            osp.join(subj_dir, 'bvecs'))
        bvals = (np.round(bvals/1000)*1000).astype(np.int)

        dwi_img = nib.load(osp.join(subj_dir, 'data.nii.gz'))
        dwi = dwi_img.get_fdata()
        min_val = np.min(dwi)
        max_val = np.max(dwi)

        dwi = 2*(dwi - min_val) / (max_val - min_val) -1
        
        b0_volumes, b0_vecs = __get_bvolumes_bvecs(0, bvals, dwi, bvecs)
        b1000_volumes, b1000_vecs = __get_bvolumes_bvecs(1000, bvals, dwi, bvecs)
        # b2000_volumes, b2000_vecs = __get_bvolumes_bvecs(2000, bvals, dwi, bvecs)
        # b3000_volumes, b3000_vecs = __get_bvolumes_bvecs(3000, bvals, dwi, bvecs)

        num_volumes = b1000_volumes.shape[0]
        for volume_idx in range(num_volumes):

            # print('Volume', volume_idx)

            b1000_info = torch.FloatTensor(6, crop_size, crop_size).zero_()
            bvec = b1000_vecs[volume_idx,...]
            b1000_info[0,...] = 1 # one-hot encode that it's b1000 for now
            b1000_info[3,...] = bvec[0]
            b1000_info[4,...] = bvec[1]
            b1000_info[5,...] = bvec[2]

            corresponding_b1000 = b1000_volumes[volume_idx, ...]
            corresponding_b1000 = np.expand_dims(corresponding_b1000, axis=0)

            sliceIdx_volume = np.ones(corresponding_b1000.shape)
            for slice in range(corresponding_b1000.shape[-1]):
                sliceIdx_volume[...,slice] *= slice

            # num_volumes = b2000_volumes.shape[0]
            # b2000_info = torch.FloatTensor(num_volumes, 6).zero_()
            # for volume_idx in range(num_volumes): 
            #     bvec = b2000_vecs[volume_idx,...]
            #     b2000_info[volume_idx,1] = 1 # one-hot encode that it's b1000 for now
            #     b2000_info[volume_idx,3] = bvec[0]
            #     b2000_info[volume_idx,4] = bvec[1]
            #     b2000_info[volume_idx,5] = bvec[2]

            # num_volumes = b3000_volumes.shape[0]
            # b3000_info = torch.FloatTensor(num_volumes, 6).zero_()
            # for volume_idx in range(num_volumes): 
            #     bvec = b3000_vecs[volume_idx,...]
            #     b3000_info[volume_idx,2] = 1 # one-hot encode that it's b1000 for now
            #     b3000_info[volume_idx,3] = bvec[0]
            #     b3000_info[volume_idx,4] = bvec[1]
            #     b3000_info[volume_idx,5] = bvec[2]
            
            subject = Subject(subj_id=subj_id, b0=ScalarImage(tensor=b0_volumes),
                            b1000=ScalarImage(tensor=corresponding_b1000), b1000_info=b1000_info,
                            slice_volume=ScalarImage(tensor=sliceIdx_volume),
                            bvec_volume=volume_idx,
                            max_slices = b0_volumes.shape[-1], 
                            num_bvec = num_volumes, dwi_affine = dwi_img.affine,
                            max_val = max_val, min_val = min_val)
                            
                            # b2000=ScalarImage(tensor=b2000_volumes), b2000_info=b2000_info,
                            # b3000=ScalarImage(tensor=b3000_volumes), b3000_info=b3000_info)
            yield subject
