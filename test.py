"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import os.path as osp
from collections import OrderedDict

from nibabel.parrec import vol_is_full

import data
from options.test_options import TestOptions
from models.s2ms_model import S2MSModel
from util.visualizer import Visualizer
from util import html
import numpy as np
import nibabel as nib
import h5py

opt = TestOptions().parse()
opt.name= 'b0_n_10xb1000_to_b2000_normBval' #'b0_to_b1000'
opt.results_dir = osp.join("/data/s2ms/results", opt.name)
opt.semantic_nc = 10 # just b0 volume
opt.label_nc = 10
opt.output_nc = 1
opt.dataset_mode = 's2ms'
opt.dataroot = '/data/s2ms/test'
opt.data_file = '/data/s2ms/test/test_2_norm_bVal.hdf5'
opt.crop_size = 256
opt.aspect_ratio = 1
opt.batchSize = 1
opt.samples_per_volume = 145 # number of Z slices
opt.subj_number = 15

dataloader = data.create_dataloader(opt)

opt.which_epoch = 4
model = S2MSModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
#web_dir = os.path.join(opt.results_dir, opt.name,
#                       '%s_%s' % (opt.phase, opt.which_epoch))
#webpage = html.HTML(web_dir,
#                    'Experiment = %s, Phase = %s, Epoch = %s' %
#                    (opt.name, opt.phase, opt.which_epoch))

# test

prev_subject = None

subj_id = None
affine = None

subj_to_volume = {}
subj_2_slice_number = {}

counter = 0


with h5py.File(osp.join(opt.results_dir, f'results_ep{opt.which_epoch}.hdf5'), 'w') as data_file:
    for i, data_i in enumerate(dataloader):

        generated = model(data_i, mode='inference')
        generated = generated.cpu().numpy().squeeze()

        subj_id = data_i['subj_id'].cpu().numpy()[0]
        bvec = data_i['b2000_bvec_val'].cpu().numpy().squeeze()
        b1000 = data_i['b1000_dwi'].cpu().numpy().squeeze()
        if f'res_{subj_id}_b2000' not in data_file:
            data_file.create_dataset(f'res_{subj_id}_b2000', data=np.expand_dims(generated, axis=0),
                                                maxshape=(None, generated.shape[0], generated.shape[1]))
            data_file.create_dataset(f'res_{subj_id}_b1000', data=np.expand_dims(b1000, axis=0),
                                                maxshape=(None, b1000.shape[0], b1000.shape[1]))
            data_file.create_dataset(f'res_{subj_id}_bvec_b2000', data=np.expand_dims(bvec, axis=0),
                                                maxshape=(None, bvec.shape[0]))
        else:
            current_slice_num = data_file[f'res_{subj_id}_b2000'].shape[0]
            data_file[f'res_{subj_id}_b2000'].resize((current_slice_num+1, generated.shape[0], generated.shape[1]))
            data_file[f'res_{subj_id}_b2000'][-1:,:,:] = generated

            current_slice_num = data_file[f'res_{subj_id}_b1000'].shape[0]
            data_file[f'res_{subj_id}_b1000'].resize((current_slice_num+1, b1000.shape[0], b1000.shape[1]))
            data_file[f'res_{subj_id}_b1000'][-1:,:,:] = b1000


            current_slice_num = data_file[f'res_{subj_id}_bvec_b2000'].shape[0]
            data_file[f'res_{subj_id}_bvec_b2000'].resize((current_slice_num+1, bvec.shape[0]))
            data_file[f'res_{subj_id}_bvec_b2000'][-1:,:] = bvec

    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_b0', data_i['b0'][b]),
    #                            ('synthesized_image', generated[b])])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1])
    # print("processing slice {}/{}".format(slice_idx, bvec_idx))


#webpage.save()
