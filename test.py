"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

from nibabel.parrec import vol_is_full

import data
from options.test_options import TestOptions
from models.s2ms_model import S2MSModel
from util.visualizer import Visualizer
from util import html
import numpy as np
import nibabel as nib

opt = TestOptions().parse()
opt.results_dir = "/data/s2ms/results"
opt.name='label2coco'
opt.semantic_nc=7 # just b0 volume
opt.label_nc = 7
opt.output_nc = 1
opt.dataset_mode = 's2ms'
opt.dataroot = '/data/s2ms/test'
opt.crop_size = 128
opt.aspect_ratio = 1
opt.batchSize = 1
opt.samples_per_volume = 145 # number of Z slices

dataloader = data.create_dataloader(opt)

model = S2MSModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test

prev_subject = None

subj_id = None
affine = None

counter = 0

for i, data_i in enumerate(dataloader):
    # print('Suject', data_i["subj_id"], '\tVolume', data_i["bvec_volume"])
    # assemble images
    if prev_subject is None or prev_subject != data_i['subj_id'][0]:
        if prev_subject is not None:
            img_path = "{}/{}_gen.nii.gz".format(opt.results_dir, data_i['subj_id'][0])
            generated_img = nib.Nifti1Image(subj_volume, data_i['dwi_affine'])
            nib.save(generated_img, img_path)

        subj_id = data_i['subj_id'][0]
        affine = np.squeeze(data_i['dwi_affine'].cpu().numpy())

        min_val = data_i['min_val'].cpu().numpy()[0]
        max_val = data_i['max_val'].cpu().numpy()[0]

        prev_subject = data_i["subj_id"][0]
        subj_volume = np.zeros((opt.crop_size, opt.crop_size,
                                 data_i["max_slices"], data_i["num_bvec"]+1))

    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    slice_idx = int(data_i['slice_volume']['data'][0,0,0,0].cpu().numpy())
    bvec_idx = int(data_i['bvec_volume'].cpu().numpy())


    subj_volume[..., slice_idx, bvec_idx+1] = np.squeeze(generated.cpu().numpy())
    subj_volume[..., slice_idx, 0] = np.squeeze(data_i['b0']['data'].cpu().numpy())

    counter += 1

    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_b0', data_i['b0'][b]),
    #                            ('synthesized_image', generated[b])])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1])
    print("processing slice {}/{}".format(slice_idx, bvec_idx))

print(counter)
img_path = "{}/{}_gen.nii.gz".format(opt.results_dir, subj_id)
subj_volume = (subj_volume + 1) * (max_val - min_val) / 2 + min_val
generated_img = nib.Nifti1Image(subj_volume, affine)
nib.save(generated_img, img_path)

webpage.save()
