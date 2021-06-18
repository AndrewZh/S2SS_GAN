from data.base_dataset import BaseDataset
from data.UniformSlice3DSampler import UniformSlice3DSampler
from data.subject_generator import subject_generator
from torchio import Queue, SubjectsDataset
from torchio.data import UniformSampler
from PIL import Image
import util.util as util
import os
import os.path as osp

class S2MSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--data_dir', type=str)
        parser.add_argument('--patch_size', type=int, default=(128,128,1), help=' patch size')
        parser.add_argument('--max_queue_len', type=int)
        parser.add_argument('--max_length', type=int, default=2000, help=' maximum length of the queue')
        # parser.add_argument('--samples_per_volume', type=int)
        return parser

    def initialize(self, opt):
        self.opt = opt

        subjects_gen = subject_generator(opt.dataroot, opt.patch_size[0])
        subjects = [s for s in subjects_gen]
        self.dataset_size = len([f for f in os.listdir(opt.dataroot) if osp.isdir(osp.join(opt.dataroot, f))])
        self.total_patch_number = self.dataset_size * opt.samples_per_volume

        subjects_dataset = SubjectsDataset(subjects,
                                transform=None)

        sampler = UniformSlice3DSampler(opt.patch_size, opt.samples_per_volume)
        self.subject_queue = Queue(subjects_dataset=subjects_dataset,
                                    max_length=opt.max_length,
                                    samples_per_volume=opt.samples_per_volume,
                                    sampler=sampler
                                    )
        
    def __getitem__(self, index):
        return self.subject_queue.__getitem__(index)

    def __len__(self):
        return self.total_patch_number
