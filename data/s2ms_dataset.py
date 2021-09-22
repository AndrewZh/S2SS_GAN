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
        parser.add_argument('--max_length', type=int, default=5000, help=' maximum length of the queue')
        # parser.add_argument('--samples_per_volume', type=int)
        return parser

    def initialize(self, opt):
        self.opt = opt

        is_test = self.opt.phase == 'test'
        self.is_shuffled = not is_test
        if not self.is_shuffled:
            self.subj_slice_counter = None

        subjects_gen = subject_generator(opt.dataroot, opt.patch_size[0], opt.subj_number)
        self.subjects = [s for s in subjects_gen]
        
        # self.dataset_size = len([f for f in os.listdir(opt.dataroot) if osp.isdir(osp.join(opt.dataroot, f))])
        # self.total_patch_number = self.dataset_size * opt.samples_per_volume * 90

        # each subject is actually b-volume
        self.total_patch_number = np.sum([s['slices_per_volume'] for s in self.subjects])
        print('Total patch number', self.total_patch_number)

        self.sampler = UniformSlice3DSampler(opt.patch_size, opt.samples_per_volume, self.is_shuffled, is_test)
        # q_max_length = len(subjects) * opt.samples_per_volume
        # self.total_patch_number =  int(opt.max_length * len(subjects) / 90) 

        # subjects_dataset = SubjectsDataset(self.subjects,
        #                         transform=None)

        # self.subject_queue = Queue(subjects_dataset=subjects_dataset,
        #                             max_length=q_max_length,
        #                             samples_per_volume=opt.samples_per_volume,
        #                             sampler=self.sampler, shuffle_subjects=is_shuffled,
        #                             shuffle_patches=is_shuffled, verbose=True
        #                             )
        
    def __getitem__(self, index):
        assert self.subjects
        if self.is_shuffled:
            np.random.shuffle(self.subjects)
        subject = self.subjects[0]
        self.subject_gen = self.sampler(subject)
        try:
            new_item = next(self.subject_gen)
        except StopIteration:
            self.subjects.pop(0)
            subject = self.subjects[0]
            self.subject_gen = self.sampler(subject)
            new_item = next(self.subject_gen)
        return new_item

    def __len__(self):
        return self.total_patch_number
