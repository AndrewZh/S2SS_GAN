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


    def __getitem__(self, index):
        data_file = h5py.File(self.opt.data_file, 'r')
        
        num_subjects = data_file['{}_subj_ids'.format(self.opt.phase)].shape[0]
        subj_idx = np.random.randint(0, num_subjects)
        subj_id = data_file['{}_subj_ids'.format(self.opt.phase)][subj_idx]
        
        num_slices = data_file['{}_{}_b0'.format(self.opt.phase, subj_id)].shape[0]
        slice_idx = np.random.randint(0, num_slices)
        
        num_b1000_vols = data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        b1000_vol_idx = np.random.randint(0, num_b1000_vols)

        num_b2000_vols = data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        b2000_vol_idx = np.random.randint(0, num_b2000_vols)

        #num_b3000_vols = data_file['{}_{}_b3000'.format(self.opt.phase, subj_id)].shape[0] // num_slices
        #b3000_vol_idx = np.random.randint(0, num_b3000_vols)

        slice_b0 = np.expand_dims(data_file['{}_{}_b0'.format(self.opt.phase, subj_id)][slice_idx].transpose(1, 0), axis=0)
        slice_b1000 = np.expand_dims(data_file['{}_{}_b1000'.format(self.opt.phase, subj_id)][b1000_vol_idx*num_slices+slice_idx].transpose(1, 0), axis=0)
        slice_b2000 = np.expand_dims(data_file['{}_{}_b2000'.format(self.opt.phase, subj_id)][b2000_vol_idx*num_slices+slice_idx].transpose(1, 0), axis=0)
        #slice_b3000 = data_file['{}_{}_b3000'.format(self.opt.phase, subj_id)][b3000_vol_idx*num_slices+slice_idx].transpose(0, 2, 1)
    
        b1000_vec = np.expand_dims(data_file['{}_{}_bvec_b1000'.format(self.opt.phase, subj_id)][b1000_vol_idx*num_slices+slice_idx], axis=0)
        b2000_vec = np.expand_dims(data_file['{}_{}_bvec_b2000'.format(self.opt.phase, subj_id)][b2000_vol_idx*num_slices+slice_idx], axis=0)

        slice_b0 = np.clip(slice_b0, 0, 1)

        return_dict = dict()
        return_dict['b0'] = slice_b0.astype(np.float32)
        return_dict['b1000_dwi'] = slice_b1000.astype(np.float32)
        return_dict['b2000_dwi'] = slice_b2000.astype(np.float32)
        return_dict['b1000_bvec_val'] = b1000_vec.astype(np.float32)
        return_dict['b2000_bvec_val'] = b2000_vec.astype(np.float32)

        return return_dict

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
        data_file = h5py.File(self.opt.data_file, 'r')
        num_items = data_file['{}_total_slices'.format(self.opt.phase)][0]
        data_file.close()
        return num_items
