from data.base_dataset import BaseDataset
from data.UniformSlice3DSampler import UniformSlice3DSampler
from data.subject_generator import subject_generator
from torchio import Queue, SubjectsDataset
from torchio.data import UniformSampler
from PIL import Image
import util.util as util
import os
import os.path as osp
import numpy as np
import h5py
import torch
from typing import TypeVar

T_co = TypeVar('T_co', covariant=True)


class S2MSDataset(BaseDataset):
    def __init__(self):
        super(S2MSDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--data_dir', type=str)
        parser.add_argument('--patch_size', type=int, default=(256,256,1), help=' patch size')
        parser.add_argument('--max_queue_len', type=int)
        # set default as 90 (num b-vectors) * 145 (num z-slices)
        parser.add_argument('--max_length', type=int, default=5000, help=' maximum length of the queue')
        # parser.add_argument('--samples_per_volume', type=int)
        return parser

    def initialize(self, opt):
        self.opt = opt

        is_test = self.opt.phase == 'test'
        self.is_shuffled = not is_test
        if not self.is_shuffled:
            self.subj_slice_counter = None

        self.iteration_index = 0

        data_file = h5py.File(self.opt.data_file, 'r')
        self.subjects = data_file['{}_subj_ids'.format(self.opt.phase)][:].tolist()

        if is_test:
            self.prepare_test_vol_indices(data_file)
        else:
            self.prepare_train_vol_indices(data_file)

        data_file.close()

    def prepare_test_vol_indices(self, data_file):
        _subjects = np.array([])
        _b1000_inds = np.array([])
        _b2000_inds = np.array([])
        _slice_inds = np.array([])
        for subj_id in self.subjects:
            num_slices = data_file['{}_{}_b0'.format(self.opt.phase, subj_id)].shape[0]
            num_b2000_vols = data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
            num_b1000_vols = data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)].shape[0] // num_slices

            b2000_inds, slice_inds = np.meshgrid(range(num_b2000_vols), range(num_slices), indexing='ij')
            b1000_inds = np.random.randint(0, high=10, size=np.product(b2000_inds.shape))

            subj_inds = np.repeat([subj_id], np.product(b2000_inds.shape))

            _subjects = np.concatenate((_subjects, subj_inds))
            _b2000_inds = np.concatenate((_b2000_inds, b2000_inds.flatten()))
            _b1000_inds = np.concatenate((_b1000_inds, b1000_inds.flatten()))
            _slice_inds = np.concatenate((_slice_inds, slice_inds.flatten()))

            del b2000_inds, slice_inds, subj_inds

        self.subj_vol_slice_indices = np.vstack((_subjects, _b1000_inds, _b2000_inds, _slice_inds)).T.astype(np.int)

    
    def prepare_train_vol_indices(self, data_file):
        _subjects = np.array([])
        _b1000_inds = np.array([])
        _b2000_inds = np.array([])
        _slice_inds = np.array([])
        for subj_id in self.subjects:
            num_slices = data_file['{}_{}_b0'.format(self.opt.phase, subj_id)].shape[0]
            num_b2000_vols = data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
            num_b1000_vols = data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)].shape[0] // num_slices

            # take first 10 as first N b-vectors are uniformly distributed
            b1000_inds, b2000_inds, slice_inds = np.meshgrid(list(range(10)), range(num_b2000_vols), range(num_slices), indexing='ij')
            subj_inds = np.repeat([subj_id], len(b1000_inds.flatten()))
            
            _subjects = np.concatenate((_subjects, subj_inds))
            _b1000_inds = np.concatenate((_b1000_inds, b1000_inds.flatten()))
            _b2000_inds = np.concatenate((_b2000_inds, b2000_inds.flatten()))
            _slice_inds = np.concatenate((_slice_inds, slice_inds.flatten()))

            del b1000_inds, b2000_inds, slice_inds, subj_inds

        self.subj_vol_slice_indices = np.vstack((_subjects, _b1000_inds, _b2000_inds, _slice_inds)).T.astype(np.int)
        if self.is_shuffled:
            np.random.shuffle(self.subj_vol_slice_indices)
        
    def train_data_load(self, data_file, index):
       
        (subj_id, b1000_vol_idx, b2000_vol_idx, slice_idx) = self.subj_vol_slice_indices[index,...]
        
        #num_subjects = data_file['{}_subj_ids'.format(self.opt.phase)].shape[0]
        #subj_idx = np.random.randint(0, num_subjects)
        #subj_id = data_file['{}_subj_ids'.format(self.opt.phase)][subj_idx]
        
        num_slices = data_file['{}_{}_b0'.format(self.opt.phase, subj_id)].shape[0]
        #slice_idx = np.random.randint(0, num_slices)
        
        #num_b1000_vols = data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        #b1000_vol_idx = np.random.randint(0, num_b1000_vols)

        #num_b2000_vols = data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        #b2000_vol_idx = np.random.randint(0, num_b2000_vols)

        #num_b3000_vols = data_file['{}_{}_b3000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        #b3000_vol_idx = np.random.randint(0, num_b3000_vols)

        slice_b0 = np.expand_dims(data_file['{}_{}_b0'.format(self.opt.phase, subj_id)][slice_idx].transpose(1, 0), axis=0)
        slice_b1000 = np.expand_dims(data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)][b1000_vol_idx*num_slices+slice_idx].transpose(1, 0), axis=0)
        slice_b2000 = np.expand_dims(data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)][b2000_vol_idx*num_slices+slice_idx].transpose(1, 0), axis=0)
        #slice_b3000 = data_file['{}_{}_b3000'.format(self.opt.phase, subj_id)][b3000_vol_idx*num_slices+slice_idx].transpose(0, 2, 1)
    
        b1000_vec = np.expand_dims(data_file['{}_{}_bvec_b1000'.format(self.opt.phase, subj_id)][b1000_vol_idx*num_slices+slice_idx], axis=0)
        b2000_vec = np.expand_dims(data_file['{}_{}_bvec_b2000'.format(self.opt.phase, subj_id)][b2000_vol_idx*num_slices+slice_idx], axis=0)

        if np.random.uniform(0,1) <= 0.5:
            b1000_vec[:,:3] *= -1 
        if np.random.uniform(0,1) <= 0.5:
            b2000_vec[:,:3] *= -1 
        slice_b0 = np.clip(slice_b0, 0, 1)

        return_dict = dict()
        return_dict['b0'] = slice_b0.astype(np.float32)
        return_dict['b1000_dwi'] = slice_b1000.astype(np.float32)
        return_dict['b2000_dwi'] = slice_b2000.astype(np.float32)
        return_dict['b1000_bvec_val'] = b1000_vec.astype(np.float32)
        return_dict['b2000_bvec_val'] = b2000_vec.astype(np.float32)
        return_dict['subj_id'] = subj_id
        
        return return_dict

    def test_data_load(self, data_file, index):
        subjects = data_file['{}_subj_ids'.format(self.opt.phase)]
        
        target_subject = None
        for subj_id in subjects:
            b2000_vecs = data_file['{}_{}_bvec_b2000'.format(self.opt.phase, subj_id)]
            num_b2000_slices = b2000_vecs.shape[0]

            if index >= num_b2000_slices:
                index -= num_b2000_slices
            else:
                target_subject = subj_id
                break

        slice_idx = index
        
        num_slices = data_file['{}_{}_b0'.format(self.opt.phase, target_subject)].shape[0]
        
        num_b1000_vols = data_file['{}_{}_b1000'.format(self.opt.phase, target_subject)].shape[0] // num_slices
        b1000_vol_idx = np.random.randint(0, num_b1000_vols)

        slice_in_volume = slice_idx % num_slices
        slice_b0 = np.expand_dims(data_file['{}_{}_b0'.format(self.opt.phase, target_subject)][slice_in_volume].transpose(1, 0), axis=0)
        slice_b1000 = np.expand_dims(data_file['{}_{}_b1000'.format(self.opt.phase, target_subject)][b1000_vol_idx*num_slices+slice_in_volume].transpose(1, 0), axis=0)
        
        slice_b2000 = np.expand_dims(data_file['{}_{}_b2000'.format(self.opt.phase, target_subject)][slice_idx].transpose(1, 0), axis=0)
    
        b1000_vec = np.expand_dims(data_file['{}_{}_bvec_b1000'.format(self.opt.phase, target_subject)][b1000_vol_idx*num_slices+slice_in_volume], axis=0)
        b2000_vec = np.expand_dims(data_file['{}_{}_bvec_b2000'.format(self.opt.phase, target_subject)][slice_idx], axis=0)

        slice_b0 = np.clip(slice_b0, 0, 1)

        return_dict = dict()
        return_dict['b0'] = slice_b0.astype(np.float32)
        return_dict['b1000_dwi'] = slice_b1000.astype(np.float32)
        return_dict['b2000_dwi'] = slice_b2000.astype(np.float32)
        return_dict['b1000_bvec_val'] = b1000_vec.astype(np.float32)
        return_dict['b2000_bvec_val'] = b2000_vec.astype(np.float32)
        return_dict['subj_id'] = subj_id
        
        return return_dict

    def __getitem__(self, index):
        data_file = h5py.File(self.opt.data_file, 'r')
        
        if self.opt.phase == 'train':
            data = self.train_data_load(data_file, self.iteration_index)
            self.iteration_index = (self.iteration_index + 1) % self.__len__()

            return data
        else:
            #data = self.test_data_load(data_file, self.iteration_index)
            data = self.train_data_load(data_file, self.iteration_index)
            self.iteration_index += 1
            return data

    # def __getitem__(self, index):
    #     assert self.subjects
    #     if self.is_shuffled:
    #         np.random.shuffle(self.subjects)
    #     subject = next(self.subjects_gen) #self.subjects[0]
    #     self.subject_slice_gen = self.sampler(subject)
    #     try:
    #         new_item = next(self.subject_slice_gen)
    #     except StopIteration:
    #         self.subjects.pop(0)
    #         subject = next(self.subjects_gen) # self.subjects[0]
    #         self.subject_slice_gen = self.sampler(subject)
    #         new_item = next(self.subject_slice_gen)
    #     return new_item

    def __len__(self):
        #data_file = h5py.File(self.opt.data_file, 'r')

        #num_items = data_file['{}_total_slices'.format(self.opt.phase)][0] // 3 # approximately to make up for the error in hdf file generation
        #data_file.close()
        return self.subj_vol_slice_indices.shape[0]
