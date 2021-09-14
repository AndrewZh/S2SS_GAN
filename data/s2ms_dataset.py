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


class S2MSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--data_dir', type=str)
        parser.add_argument('--patch_size', type=int, default=(256,256,1), help=' patch size')
        parser.add_argument('--max_queue_len', type=int)
        # set default as 90 (num b-vectors) * 145 (num z-slices)
        parser.add_argument('--max_length', type=int, default=100000, help=' maximum length of the queue')
        # parser.add_argument('--samples_per_volume', type=int)
        return parser

    def initialize(self, opt):
        self.opt = opt

        is_test = self.opt.phase == 'test'
        is_shuffled = not is_test

        subjects_gen = subject_generator(opt.dataroot, opt.patch_size[0], opt.subj_number)
        subjects = [s for s in subjects_gen]
        # self.dataset_size = len([f for f in os.listdir(opt.dataroot) if osp.isdir(osp.join(opt.dataroot, f))])
        # self.total_patch_number = self.dataset_size * opt.samples_per_volume * 90

        # each subject is actually b-volume
        self.total_patch_number = np.sum([s['slices_per_volume'] for s in subjects])
        print('Total patch number', self.total_patch_number)
        q_max_length = len(subjects) * opt.samples_per_volume
        # self.total_patch_number =  int(opt.max_length * len(subjects) / 90) 

        subjects_dataset = SubjectsDataset(subjects,
                                transform=None)

        self.sampler = UniformSlice3DSampler(opt.patch_size, opt.samples_per_volume, is_shuffled, is_test)
        self.subject_queue = Queue(subjects_dataset=subjects_dataset,
                                    max_length=q_max_length,
                                    samples_per_volume=opt.samples_per_volume,
                                    sampler=self.sampler, shuffle_subjects=is_shuffled,
                                    shuffle_patches=is_shuffled, verbose=True
                                    )
        
    def __getitem__(self, index):
        return self.subject_queue.__getitem__(index)

    def __len__(self):
        return self.total_patch_number

    def total_generated_patches(self):
        return self.sampler.patches_generated