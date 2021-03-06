import numpy as np
import h5py
import nibabel as nib
import os
import os.path as osp
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import normalize_data


def save_bShell(data_file, bKey, subject, vols, bvec, slice_mask, target_mode):
    added_slices = 0
    for vol_idx in range(vols.shape[-1]):
        print('{} {}:\t{} / {}'.format(subject, bKey, vol_idx, vols.shape[-1]), end='\r')
        
        vol = vols[..., vol_idx]
        
        vol_bvecs = np.repeat(np.expand_dims(bvec[vol_idx,:], axis=0),
                                    repeats=vol.shape[2], axis=0)
        vol_bvecs = vol_bvecs[slice_mask,...]
        vol_bvecs = vol_bvecs.astype(np.float32)

        vol = np.moveaxis(vol, 2, 0)
        vol = vol[slice_mask, ...]
        vol = vol.astype(np.float32)

        added_slices += vol.shape[0]
        
        if f'{target_mode}_{subject}_{bKey}' not in data_file:
            data_file.create_dataset(f'{target_mode}_{subject}_{bKey}', data=vol,
                                                maxshape=(None, vol.shape[1], vol.shape[2]))
            
            data_file.create_dataset(f'{target_mode}_{subject}_bvec_{bKey}', data=vol_bvecs,
                                                maxshape=(None, vol_bvecs.shape[1]))
        else:
            current_slice_num = data_file[f'{target_mode}_{subject}_{bKey}'].shape[0]
            data_file[f'{target_mode}_{subject}_{bKey}'].resize((current_slice_num+vol.shape[0], vol.shape[1], vol.shape[2]))
            data_file[f'{target_mode}_{subject}_{bKey}'][-vol.shape[0]:,:,:] = vol

            current_slice_num = data_file[f'{target_mode}_{subject}_bvec_{bKey}'].shape[0]
            data_file[f'{target_mode}_{subject}_bvec_{bKey}'].resize((current_slice_num+vol_bvecs.shape[0], vol_bvecs.shape[1]))
            data_file[f'{target_mode}_{subject}_bvec_{bKey}'][-vol_bvecs.shape[0]:,:] = vol_bvecs
    print('')
    return added_slices

def preprocess_struct_MRI(mri):
    mri /= mri.max() 
    return mri

def preprocess_DWI(mri, b0_mask):
    dwi_mask = not b0_mask
    dwi_vols = mri[..., dwi_mask]

    norm_mri = normalize_data(mri, b0_mask)
    norm_dwi = norm_mri[..., dwi_mask]
    norm_dwi[dwi_vols==0] = 0
    norm_dwi = np.clip(norm_dwi, 0, 1)

    b0_vols = mri[..., b0_mask]
    b0_avrg_vol = np.mean(b0_vols, axis=-1)
    b0_max = b0_avrg_vol.max()
    b0_avrg_vol = b0_avrg_vol / b0_max
    b0_avrg_vol[b0_avrg_vol < 0] = 0

    return b0_avrg_vol, norm_dwi

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

def nii2hdf5(hcp_dir, subjects, target_mode):    
    # dwi_data = None
    # bval_bvec_data = None
    # t1_data = None
    # t2_data = None
    # b0_data = None

    # target_mode = 'train'
    print(target_mode)
    total_slices = 0
    subjects = list(map(int, subjects))
    with h5py.File(osp.join(hcp_dir, f'{target_mode}_{len(subjects)}_norm_bVal.hdf5'), 'w') as data_file:    
        data_file.create_dataset(f'{target_mode}_subj_ids', data=np.array(subjects))
        for i, subject in enumerate(subjects):
            print(subject)
            # if i > 6:
            #     target_mode = 'val'
            dwi = nib.load(osp.join(hcp_dir, str(subject), 'data_padded.nii.gz')).get_fdata()
            print('Loaded data')
            mask = nib.load(osp.join(hcp_dir, str(subject), 'nodif_brain_mask.nii.gz')).get_fdata()
            padding = get_padding(*mask.shape)
            mask = np.pad(mask, padding, mode='constant')
            dwi[mask==0,:] = 0
            bval, bvec = read_bvals_bvecs(osp.join(hcp_dir, str(subject), 'bvals'),
                                        osp.join(hcp_dir, str(subject), 'bvecs'))
            # bval[bval < 10] = 0
            rbval = (np.round(bval/1000)*1000).astype(np.int32)

            # NORMALIZE B-VALUES
            max_bval = np.max(rbval)
            bval = bval /max_bval
            bval_bvec = np.concatenate((bvec, np.expand_dims(bval,axis=1)),axis=1)

            b0_mask = (rbval == 0)
            dwi_mask = rbval > 0
            dwi_rbvals = rbval[dwi_mask]
            dwi_bval_bvec = bval_bvec[dwi_mask,:]
            b1000_mask = (dwi_rbvals == 1000)
            b2000_mask = (dwi_rbvals == 2000)
            b3000_mask = (dwi_rbvals == 3000)

            dwi_vols = dwi[..., dwi_mask]            

            b0_vols = dwi[..., b0_mask]

            norm_dwi = normalize_data(dwi, b0_mask)[..., dwi_mask]
            print('Normalized')
            norm_dwi[dwi_vols==0] = 0
            norm_dwi = np.clip(norm_dwi, 0, 1)

            b1000_vols = norm_dwi[..., b1000_mask]
            b2000_vols = norm_dwi[..., b2000_mask]
            b3000_vols = norm_dwi[..., b3000_mask]

            dwi_bval_bvec[:,3] /= np.max(bval) # 3000 in our case
            bvec_b1000 = dwi_bval_bvec[b1000_mask,:]
            bvec_b2000 = dwi_bval_bvec[b2000_mask,:]
            bvec_b3000 = dwi_bval_bvec[b3000_mask,:]


            b0_avrg_vol = np.mean(b0_vols[...], axis=-1)
            b0_max = b0_avrg_vol.max()
            b0_avrg_vol = b0_avrg_vol / b0_max
            b0_avrg_vol[b0_avrg_vol < 0] = 0

            brain_mask = b0_avrg_vol > 0
            flat_slices = np.reshape(brain_mask,(-1, brain_mask.shape[-1]))
            slice_mask = np.sum(flat_slices, axis=0) > 0

            b0_vol = b0_avrg_vol
            b0_vol = np.moveaxis(b0_vol, 2, 0)
            b0_vol = b0_vol[slice_mask, ...]
            b0_vol = b0_vol.astype(np.float32)

            data_file.create_dataset(f'{target_mode}_{subject}_b0', data=b0_vol, 
                                        maxshape=(None, b0_vol.shape[1], b0_vol.shape[2]))

            print('Saving shells')
            save_bShell(data_file, 'b1000', subject, b1000_vols, bvec_b1000, slice_mask, target_mode)
            total_slices += save_bShell(data_file, 'b2000', subject, b2000_vols, bvec_b2000, slice_mask, target_mode)
            save_bShell(data_file, 'b3000', subject, b3000_vols, bvec_b3000, slice_mask, target_mode)
        data_file.create_dataset(f'{target_mode}_total_slices', data=np.array([total_slices]))


if __name__ == "__main__":
    # data_dir = '/data/s2ms/train' 
    # # subjects = [s for s in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, s))]

    # subjects = ['601127', '622236', '644044', '645551', 
    #             '654754', '704238', '715041', '748258', '761957',
    #             '784565', '792564', '814649', '837560', '857263', '859671']

    # nii2hdf5(data_dir, subjects[:15], 'train')

    data_dir = '/data/s2ms/test' 
    # subjects = [s for s in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, s))]
    subjects = ['872158',  '872764']# '889579',  '894673',  '901442',  '910241',  '912447',  '922854',  '930449',
                #'932554',  '958976',  '978578',  '979984',  '983773',  '991267']

    nii2hdf5(data_dir, subjects, 'test')
