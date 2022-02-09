import h5py
import nibabel as nib
import numpy as np
import os.path as osp
import os
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import normalize_data


def extract_subjects(data_keys):
    subjects = set()

    for k in data_keys:
        if not k.endswith('b1000'):
            continue
        s = k[len('res_'):-len('_b1000')]
        subjects.add(s)

    return subjects

def get_slice_mask(mask_path):
    brain_mask = nib.load(mask_path).get_fdata()
    flat_slices = np.reshape(brain_mask,(-1, brain_mask.shape[-1]))
    slice_mask = np.sum(flat_slices, axis=0) > 0  

    return slice_mask

def get_padding(x, y, z):
    """
    perform zero padding on the mask data
    if for the difference between size and the shape of the mask is odd,
    right and up directions are padded with one more zero
    change the order of right, left or down, up  if the opposite is preferred.
    """
    target_size = 256
    if (target_size - x) % 2 == 0:

        left = (target_size - x) // 2
        right = (target_size - x) // 2
    else:
        left = (target_size - x) // 2
        right = 1 + (target_size - x) // 2
    if (target_size - y) % 2 == 0:

        down = (target_size - y) // 2
        up = (target_size - y) // 2
    else:
        down = (target_size - y) // 2
        up = 1 + (target_size - y) // 2
    return [(right, left), (down, up), (0, 0)]


def write_data(out_dir, subj_id, img, bvec, bval):
    if not osp.isdir(f'{out_dir}/{subj_id}'):
        os.mkdir(f'{out_dir}/{subj_id}')

    nib.save(img, f'{out_dir}/{subj_id}/data.nii.gz')

    with open(f'{out_dir}/{subj_id}/bvals', 'w') as bval_file:
        bval_file.write(' '.join(map(str, bval.tolist())))
        bval_file.write('\n')
    
    with open(f'{out_dir}/{subj_id}/bvecs', 'w') as bvec_file:
        for i in range(bvec.shape[1]):
            bvec_file.write(' '.join(map(str, bvec[:,i].tolist())))
            bvec_file.write('\n')


def extract_dwi(hdf_filename, ref_hdf5, subj_id, out_dir):
    dwi_key = f'res_{subj_id}_b2000'
    bvec_val_key = f'res_{subj_id}_bvec_b2000'

    with h5py.File(ref_hdf5, 'r') as ref_data:
        num_slices = ref_data[f'test_{subj_id}_b0'].shape[0]
    
    non_pad_nii = nib.load(f'/data/s2ms/test/{subj_id}/data.nii.gz')
    non_pad_shape = non_pad_nii.shape[:-1]
    ref_nii = nib.load(f'/data/s2ms/test/{subj_id}/data_padded.nii.gz')
    dwi = ref_nii.get_fdata()
    bval, bvec = read_bvals_bvecs(f'/data/s2ms/test/{subj_id}/bvals',   
                                    f'/data/s2ms/test/{subj_id}/bvecs')

    rbval = (np.round(bval/1000)*1000).astype(np.int32)
    b0_mask = (rbval == 0)
    b1000_mask = (rbval == 1000)
    dwi_mask = rbval > 0    
    
    b0_vols = dwi[..., b0_mask]
    b1000 = dwi[..., b1000_mask]
    # b1000[dwi[..., b1000_mask] == 0] = 0
    # b1000 = np.clip(b1000, 0, 1)
    init_vols = b0_vols #np.concatenate((b0_vols, b1000), axis=-1)
    # init_mask = np.logical_or(b0_mask, b1000_mask)
    # init_vols = np.clip(init_vols, 0, 1)

    init_bvals = bval[b0_mask] #np.concatenate((bval[b0_mask], bval[b1000_mask]))
    init_bvecs = bvec[b0_mask,:] #np.concatenate((bvec[b0_mask,:],bvec[b1000_mask,:]))
    
    slice_mask = get_slice_mask(f'/data/s2ms/test/{subj_id}/nodif_brain_mask.nii.gz')
    
    with h5py.File(hdf_filename, 'r') as data:
        dwi_data = data[dwi_key][:,:,:]
        bvec_bval = data[bvec_val_key][:]
        bvec_bval[:,3] = bvec_bval[:,3] * np.max(bval) 
    
    b2000_bvec = bvec_bval[::num_slices,:3]
    b2000_bval = bvec_bval[::num_slices,3]

    
    final_bvec = np.concatenate((init_bvecs,b2000_bvec), axis=0)
    final_bval = np.concatenate((init_bvals,b2000_bval), axis=0)
 
    print('adding volume')
    num_vols = dwi_data.shape[0] // num_slices
    dwi_vols = np.zeros((num_slices,)+dwi_data.shape[1:]+(num_vols,))
    for i in range(num_vols):
        start_idx = i*num_slices
        end_idx = (i+1) * num_slices
        dwi_vols[:,:,:,i] = dwi_data[start_idx:end_idx,...]

    dwi_vols = np.moveaxis(dwi_vols, 0, 2)
    dwi_vols = dwi_vols.transpose((1, 0, 2, 3))

    full_size_data = np.zeros((dwi_vols.shape[0], dwi_vols.shape[1], dwi.shape[2], dwi_vols.shape[-1]))
    full_size_data[:,:,slice_mask,:] = np.clip(dwi_vols,0,1)
    
    avrg_b0 = b0_vols.mean(-1)
    full_size_data *= avrg_b0[..., None]
    
    padding = get_padding(*non_pad_shape)
    (left, right) = padding[0]
    (top, bottom) = padding[1]

    final_data = np.concatenate((init_vols, full_size_data), axis=-1)
    final_data = final_data[left:-right, top:-bottom, ...]

    final_img = nib.Nifti1Image(final_data, affine=ref_nii.affine)
    
    write_data(out_dir, subj_id, final_img, final_bvec, final_bval)


if __name__ == '__main__':
    hdf_filename = '/data/s2ms/results/b0_n_10xb1000_to_b2000_normBval/results_ep4.hdf5'
    out_dir = '/data/s2ms/results/b0_n_10xb1000_to_b2000_normBval'

    with h5py.File(hdf_filename, 'r') as data:
        data_keys = list(data.keys())
        # print(data_keys)
    
    subjects = extract_subjects(data_keys)
    print('Whole set', subjects)
    
    for subj_id in subjects:
        print(subj_id)
        extract_dwi(hdf_filename, '/data/s2ms/test/test_2_norm_bVal.hdf5', subj_id, out_dir)

