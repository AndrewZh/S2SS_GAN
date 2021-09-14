"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from trainers.s2ms_trainer import S2MSTrainer
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.s2ms_trainer import S2MSTrainer
import torch

# parse options
opt = TrainOptions().parse()
print(opt.continue_train)
opt.semantic_nc=14 # just b0 volume
opt.name = 'b0_n_b1000_to_b2000'
opt.label_nc = 14
opt.output_nc = 1
opt.dataset_mode = 's2ms'
opt.dataroot = '/data/s2ms/train'
opt.crop_size = 256
opt.aspect_ratio = 1
opt.batchSize = 10
opt.samples_per_volume = 145 # num z slices
opt.subj_number = 40

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = S2MSTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            data_i['b0']['data'] = torch.squeeze(data_i['b0']['data'], 4)
            data_i['b1000']['data'] = torch.squeeze(data_i['b1000']['data'], 4)
            data_i['b2000']['data'] = torch.squeeze(data_i['b2000']['data'], 4)
            visuals = OrderedDict([('input_b0', data_i['b0']['data']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_b2000', data_i['b2000']['data'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
