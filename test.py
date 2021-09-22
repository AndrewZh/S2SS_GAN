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

opt = TestOptions().parse()
opt.results_dir = "/data/s2ms/results"
opt.name= 'b0_n_b1000_to_b2000' #'b0_to_b1000'
opt.semantic_nc = 14 # just b0 volume
opt.label_nc = 14
opt.output_nc = 1
opt.dataset_mode = 's2ms'
opt.dataroot = '/data/s2ms/test'
opt.crop_size = 256
opt.aspect_ratio = 1
opt.batchSize = 1
opt.samples_per_volume = 145 # number of Z slices
opt.subj_number = 7

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

subj_to_volume = {}
subj_2_slice_number = {}

counter = 0

for i, data_i in enumerate(dataloader):
    # print('Suject', data_i["subj_id"], '\tVolume', data_i["bvec_volume"])
    # assemble images
    # if prev_subject is None or prev_subject != data_i['subj_id'][0]:
        # if prev_subject is not None:
        #     img_path = osp.join(opt.results_dir, opt.name, f"{prev_subject}_gen.nii.gz")
        #     subj_volume = (subj_volume + 1) * (max_val - min_val) / 2 + min_val
        #     affine = np.squeeze(data_i['dwi_affine'].cpu().numpy())
        #     generated_img = nib.Nifti1Image(subj_volume, affine)
        #     print(generated_img.shape[-1])
        #     print(img_path)
        #     # nib.save(generated_img, img_path)
    subj_id = data_i['subj_id'][0]
    affine = np.squeeze(data_i['dwi_affine'].cpu().numpy())
    counter += 1

    min_val = data_i['min_val'].cpu().numpy()[0]
    max_val = data_i['max_val'].cpu().numpy()[0]

    prev_subject = data_i["subj_id"][0]
    
    num_bvec = data_i["num_bvec"].cpu().numpy()[0]
    max_slices = data_i["max_slices"].cpu().numpy()[0]

    assert (subj_id not in subj_to_volume) == (subj_id not in subj_2_slice_number)

    if subj_id not in subj_to_volume:
        subj_volume = -1 * np.ones((opt.crop_size, opt.crop_size,
                                max_slices, num_bvec+1))
        subj_to_volume[subj_id] = subj_volume
        slices_per_volume = data_i["slices_per_volume"].cpu().numpy()[0]
        subj_2_slice_number[subj_id] = num_bvec * slices_per_volume
        assert(num_bvec+1==nib.load(f'/data/s2ms/results/b0_to_b1000/{subj_id}_gt.nii.gz').shape[-1])
    # if i * opt.batchSize >= opt.how_many:
    #     break

    generated = model(data_i, mode='inference')
    slice_idx = int(data_i['slice_volume']['data'][0,0,0,0].cpu().numpy())
    bvec_idx = int(data_i['bvec_volume'].cpu().numpy())

    subj_volume = subj_to_volume[subj_id]
    subj_volume[..., slice_idx, bvec_idx+1] = np.squeeze(generated.cpu().numpy())
    subj_volume[..., slice_idx, 0] = np.squeeze(data_i['b0']['data'].cpu().numpy())
    subj_to_volume[subj_id] = subj_volume

    subj_2_slice_number[subj_id] = subj_2_slice_number[subj_id] - 1

    if subj_2_slice_number[subj_id] == 0:
        img_path = osp.join(opt.results_dir, opt.name, f"{prev_subject}_gen.nii.gz")
        subj_volume = subj_to_volume[subj_id]
        subj_volume = (subj_volume + 1) * (max_val - min_val) / 2 + min_val
        affine = np.squeeze(data_i['dwi_affine'].cpu().numpy())
        generated_img = nib.Nifti1Image(subj_volume, affine)
        nib.save(generated_img, img_path)

        subj_to_volume.pop(subj_id)

    # counter += 1

    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_b0', data_i['b0'][b]),
    #                            ('synthesized_image', generated[b])])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1])
    # print("processing slice {}/{}".format(slice_idx, bvec_idx))


assert not subj_to_volume, "Some subject not saved"

# img_path = osp.join(opt.results_dir, opt.name, f"{prev_subject}_gen.nii.gz")
# subj_volume = (subj_volume + 1) * (max_val - min_val) / 2 + min_val
# subj_volume = subj_volume.astype(np.float32)
# generated_img = nib.Nifti1Image(subj_volume, affine)
# print(generated_img.shape[-1])
# print(img_path)
# nib.save(generated_img, img_path)

webpage.save()
