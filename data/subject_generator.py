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

def subject_generator(data_dir, crop_size, subj_number) -> Generator[Subject, None, None]:
    subj_ids = [f for f in sorted(os.listdir(data_dir)) if osp.isdir(osp.join(data_dir,f))]

    # subj_ids = ['872764', '899885', '901038']
    for subj_id in subj_ids[:subj_number]: # sina: why iteration over subjects here?
        # if subj_id== '715041': # sina: just take one subject 
        subj_dir = osp.join(data_dir, subj_id)
        print(subj_id)

        bvals, bvecs = dio.read_bvals_bvecs(osp.join(subj_dir, 'bvals'),
                                            osp.join(subj_dir, 'bvecs'))
        bvals = (np.round(bvals/1000)*1000).astype(np.int)

        dwi_img = nib.load(osp.join(subj_dir, 'data_padded.nii.gz'))
        dwi = dwi_img.get_fdata()
        dwi_affine = dwi_img.affine
        min_val = np.min(dwi)
        max_val = np.max(dwi)

        dwi = 2*(dwi - min_val) / (max_val - min_val) -1
        
        b0_volumes, b0_vecs = __get_bvolumes_bvecs(0, bvals, dwi, bvecs)
        b0_volume_file = osp.join(subj_dir, "b0.nii.gz")

        flat_slices = np.reshape(b0_volumes,(-1, b0_volumes.shape[-1]))
        slice_mask = np.sum(flat_slices+1, axis=0) > 0

        if not osp.exists(b0_volume_file):
            nib.save(nib.Nifti1Image(b0_volumes[..., slice_mask], dwi_affine), b0_volume_file)

        assert(osp.exists(b0_volume_file))

        sliceIdx_file = osp.join(subj_dir, "slice_indices.nii.gz")
        # slices_per_volume = nib.load(sliceIdx_file).shape[-1] ## placeholder
        
        # assert(len(b0_volumes.shape) == 3)
        assert osp.exists(sliceIdx_file)
        if not osp.exists(sliceIdx_file):
            sliceIdx_volume = np.ones(b0_volumes.shape[1:])
            for slice in range(b0_volumes.shape[-1]):
                sliceIdx_volume[...,slice] *= slice
            sliceIdx_volume = sliceIdx_volume[..., slice_mask]
            nib.save(nib.Nifti1Image(sliceIdx_volume, dwi_affine), sliceIdx_file)

        b1000_volumes, b1000_vecs = __get_bvolumes_bvecs(1000, bvals, dwi, bvecs)
        b2000_volumes, b2000_vecs = __get_bvolumes_bvecs(2000, bvals, dwi, bvecs)

        # b3000_volumes, b3000_vecs = __get_bvolumes_bvecs(3000, bvals, dwi, bvecs)

        slices_per_volume = np.sum(slice_mask)

        del dwi
        del dwi_img

        num_volumes = b1000_volumes.shape[0]
        # num_volumes = np.sum(bvals == 1000) ## placeholder
        print('b1000 --', num_volumes)
        b2000_num_volumes = b2000_volumes.shape[0]
        for volume_idx in range(num_volumes):

            # print('Volume', volume_idx)

            b1000_info = torch.FloatTensor(6, crop_size, crop_size).zero_()
            bvec = b1000_vecs[volume_idx,...]
            b1000_info[0,...] = 1 # one-hot encode that it's b1000 for now
            b1000_info[3,...] = bvec[0]
            b1000_info[4,...] = bvec[1]
            b1000_info[5,...] = bvec[2]

            b1000_volume_file = osp.join(subj_dir, "b1000_{}.nii.gz".format(volume_idx))           
            if not osp.exists(b1000_volume_file):
                corresponding_b1000 = b1000_volumes[volume_idx, ...]
                corresponding_b1000 = corresponding_b1000[...,slice_mask]
                nib.save(nib.Nifti1Image(corresponding_b1000, dwi_affine), b1000_volume_file)
        
            # corresponding_b1000 = np.expand_dims(corresponding_b1000, axis=0)

            assert(osp.exists(b1000_volume_file))

            for b2000_volume_idx in range(b2000_num_volumes):
                b2000_info = torch.FloatTensor(6, crop_size, crop_size).zero_()
                bvec = b2000_vecs[b2000_volume_idx,...]
                b2000_info[1,...] = 1 # one-hot encode that it's b1000 for now
                b2000_info[3,...] = bvec[0]
                b2000_info[4,...] = bvec[1]
                b2000_info[5,...] = bvec[2]

                b2000_volume_file = osp.join(subj_dir, "b2000_{}.nii.gz".format(volume_idx))           
                if not osp.exists(b2000_volume_file):
                    corresponding_b2000 = b2000_volumes[b2000_volume_idx, ...]
                    corresponding_b2000 = corresponding_b2000[...,slice_mask]
                    nib.save(nib.Nifti1Image(corresponding_b2000, dwi_affine), b2000_volume_file)
            
                # corresponding_b2000 = np.expand_dims(corresponding_b2000, axis=0)

                assert(osp.exists(b2000_volume_file))

                subject = Subject(subj_id=subj_id, b0=ScalarImage(b0_volume_file),
                                b1000=ScalarImage(b1000_volume_file), b1000_info=b1000_info,
                                b2000=ScalarImage(b2000_volume_file), b2000_info=b2000_info,
                                slice_volume=ScalarImage(sliceIdx_file),
                                bvec_volume=volume_idx,
                                max_slices = b0_volumes.shape[-1], 
                                num_bvec = num_volumes, dwi_affine = dwi_affine,
                                max_val = max_val, min_val = min_val,
                                slices_per_volume = slices_per_volume)
            # subject = Subject(subj_id=subj_id, 
            #                 #b0=ScalarImage(b0_volume_file),
            #                 # b1000=ScalarImage(b1000_volume_file), b1000_info=b1000_info,
            #                 slice_volume=ScalarImage(sliceIdx_file),
            #                 bvec_volume=volume_idx,
            #                 # max_slices = b0_volumes.shape[-1], 
            #                 # num_bvec = num_volumes, dwi_affine = dwi_affine,
            #                 # max_val = max_val, min_val = min_val,
            #                 slices_per_volume = slices_per_volume)
                            
                            # b2000=ScalarImage(tensor=b2000_volumes), b2000_info=b2000_info,
                            # b3000=ScalarImage(tensor=b3000_volumes), b3000_info=b3000_info)
                yield subject
