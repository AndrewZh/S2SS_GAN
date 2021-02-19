from torchio import ScalarImage, Subject
import os.path as osp
import os
import dipy.io as dio
import numpy as np
import nibabel as nib
import torch


def __get_bvolumes_bvecs(bval, all_bvals, dwi, bvecs):
    b_mask = all_bvals == bval
    b_volumes = torch.from_numpy(np.moveaxis(dwi[:,:,:,b_mask],-1,0))
    b_vecs = torch.from_numpy(bvecs[b_mask,:])

    return b_volumes, b_vecs

def subject_generator(data_dir):
    subj_ids = os.listdir(data_dir)

    for subj_id in subj_ids:
        subj_dir = osp.join(data_dir, subj_id)

        bvals, bvecs = dio.read_bvals_bvecs(osp.join(subj_dir, 'bvals'),
                                            osp.join(subj_dir, 'bvecs'))
        bvals = (np.round(bvals/1000)*1000).astype(np.int)

        dwi = nib.load(osp.join(subj_dir, 'data.nii.gz')).get_fdata()
        
        b0_volumes, b0_vecs = __get_bvolumes_bvecs(0, bvals, dwi, bvecs)
        b1000_volumes, b1000_vecs = __get_bvolumes_bvecs(1000, bvals, dwi, bvecs)
        b2000_volumes, b2000_vecs = __get_bvolumes_bvecs(2000, bvals, dwi, bvecs)
        b3000_volumes, b3000_vecs = __get_bvolumes_bvecs(3000, bvals, dwi, bvecs)

        subject = Subject(b0=ScalarImage(tensor=b0_volumes), b0_vecs=b0_vecs,
                          b1000=ScalarImage(tensor=b1000_volumes), b1000_vecs=b1000_vecs,
                          b2000=ScalarImage(tensor=b2000_volumes), b2000_vecs=b2000_vecs,
                          b3000=ScalarImage(tensor=b3000_volumes), b3000_vecs=b3000_vecs)

        yield subject
