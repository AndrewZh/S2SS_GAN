from data.base_dataset import BaseDataset
from data.UniformSlice3DSampler import UniformSlice3DSampler
from data.subject_generator import subject_generator
from torchio import Queue, SubjectsDataset
from PIL import Image
import util.util as util
import os
import os.path as osp

class S2MSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--data_dir', type=str)
        parser.add_argument('--patch_size', type=int)
        parser.add_argument('--max_queue_len', type=int)
        # parser.add_argument('--samples_per_volume', type=int)
        return parser

    def initialize(self, opt):
        self.opt = opt

        subjects = subject_generator(opt.dataroot)
        self.dataset_size = len([f for f in os.listdir(opt.dataroot) if osp.isdir(osp.join(opt.dataroot, f))])
        self.total_patch_number = self.dataset_size * opt.samples_per_volume

        subjects_dataset = SubjectsDataset(subjects,
                                transform=None)
        self.subject_queue = Queue(subjects_dataset,
                                    opt.max_length,
                                    opt.samples_per_volume,
                                    UniformSlice3DSampler(opt.patch_size))
        
    def __getitem__(self, index):
        self.subject_queue.__getitem__(index)

    def __len__(self):
        return self.total_patch_number
