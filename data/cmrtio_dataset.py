"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import torch
import os
import torchio as tio
from torch.utils.data import DataLoader

class CmrtioDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)

        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=4)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(no_instance=True)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images', default='/data/sina/dataset/M_Ms/OpenDataset/mms/normal/ED_ES/training/vendors/Vendor_A/Mask/')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images',default= '/data/sina/dataset/M_Ms/OpenDataset/mms/normal/ED_ES/training/vendors/Vendor_A/Image/')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """
        img_list = []
        msk_list = []
        assert os.path.exists(opt.label_dir), 'list of masks  doesnt exist'
        assert os.path.exists(opt.image_dir), 'list of images doesnt exist'

        img_list = sorted(os.listdir(os.path.join(opt.image_dir)))
        msk_list = sorted(os.listdir(os.path.join(opt.label_dir)))
        assert len(img_list) == len(msk_list)

        self.img_list = img_list
        self.msk_list = msk_list

    def initialize(self, opt):
        self.opt = opt
        self.get_paths(opt)

        subjects = []
        for (images, labels) in zip(self.img_list, self.msk_list):
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(opt.image_dir,images)),
                label=tio.LabelMap(os.path.join(opt.label_dir,labels)),
            )
            subjects.append(subject)

        
        training_transform = tio.Compose([
            # tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad((opt.crop_size, opt.crop_size, 13)),
            tio.RescaleIntensity(1),
            # tio.RandomMotion(p=0.2),
            # tio.HistogramStandardization({'mri': landmarks}),
            # tio.RandomBiasField(p=0.3),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomNoise(p=0.5),
            # tio.RandomFlip(),
            # tio.OneOf({
            #     tio.RandomAffine(): 0.8,
            #     tio.RandomElasticDeformation(): 0.2,
            # }),
            # tio.OneHot(),
        ])
# TODO: use subject_generator for creating 3D subject out of DTI 4D data       
    
        self.training_set = tio.SubjectsDataset(
            subjects, transform=training_transform)

        patch_size = (opt.crop_size,opt.crop_size,1)
        samples_per_volume = 13  # chech the definition ==>  Number of patches to extract from each volume. 
        # A small number of patches ensures a large variability in the queue, but training will be slower.
        max_queue_length = 20   # chech the definition ==> Maximum number of patches that can be stored in the queue. 
        # Using a large number means that the queue needs to be filled less often, but more CPU memory is needed to store the patches.
        sampler = tio.data.UniformSampler(patch_size)
        

        self.patches_training_set = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=int(opt.nThreads),
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        size = len(self.patches_training_set)
        self.dataset_size = size
    
    def __getitem__(self, index):
        # Label Image
        data_input = self.patches_training_set[index]

        # if using instance maps
        
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = torch.squeeze(data_input['label'][tio.DATA],-1) # sina: we dont have instance map for the cmr images, puting it to zero gives me errors
            

        input_dict = {'label': torch.squeeze(data_input['label'][tio.DATA],-1),  # dataloader expect to get image tensor with dimensions of B C W H, so I removed the D dimension
                      'instance': instance_tensor,
                      'image': torch.squeeze(data_input['image'][tio.DATA],-1),
                      'path': data_input['image']['path'],
                    
                      }

        return input_dict
    
    def __len__(self):
        return self.patches_training_set.__len__()