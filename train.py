'''
Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
###############################################################################
# cmdline.py
# Defines the command line interfaces for the robrank project
# These are the entrance functions if you use python scripts under the
# bin/ or tools/ directories.
###############################################################################

import argparse
import pytorch_lightning as thl
import re
import configs
import torch as th
import gc
import psutil
import json
import itertools as it
import numpy as np
import glob
import rich
import torchvision as vision
import os
import configs
import datasets
import models
import utility
import utility.utils as utils
#c = rich.get_console()



# ag = argparse.ArgumentParser()
# ag.add_argument('-C', '--config', type=str, required=True,
#                 help='example: "sop:res18:ptripletE".')
# ag.add_argument('-g', '--gpus', type=int, default=th.cuda.device_count(),
#                 help='number of GPUs to use')
# ag.add_argument('--dp', action='store_true',
#                 help='use th.nn.DataParallel instead of distributed.')
# ag.add_argument('--do_test', action='store_true')
# ag.add_argument('-m', '--monitor', type=str, default='Validation/r@1')
# ag.add_argument('-r', '--resume', action='store_true')
# ag.add_argument('--clip', type=float, default=0.0,
#                 help='do gradient clipping by value')
# ag.add_argument('--trail', action='store_true',
#                 help='keep the intermediate checkpoints')
# ag.add_argument('--svd', action='store_true')
# ag.add_argument('-Ck', '--ckpt', type=str, required=False, default = None,
#                 help='example: logs_mnist-c2f1-ptripletE/lightning_logs/'
#                 + 'version_0/checkpoints/epoch=7.ckpt')
# ag.add_argument('-A', '--attack', type=str, required=False, default = None)
# ag.add_argument('-D', '--device', type=str, default='cuda'
#                 if th.cuda.is_available() else 'cpu')
# ag.add_argument('-v', '--verbose', action='store_true')
# ag.add_argument('-mi', '--maxiter', type=int, default=None)
# ag.add_argument('-b', '--batchsize', type=int, default=-1,
#                 help='override batchsize')
# ag.add_argument('-X', '--dumpaxd', type=str, default='',
#                 help='path to dump the adversarial examples')
# ag = ag.parse_args()

# if ag.ckpt:
#     ag.dataset, ag.model, agat.loss = re.match(
#         r'.*logs_(\w+)-(\w+)-(\w+)/.*\.ckpt', agat.ckpt).groups()
# #print(rich.panel.Panel(''.join(ag), title='RobRank::AdvRank',
# #                         style='bold magenta'))

# else:
#     # print(rich.panel.Panel(' '.join(argv), title='RobRank::Train',
# #                          style='bold magenta'))
# # c.print(vars(ag))
#     ag.dataset, ag.model, ag.loss = re.match(
#         r'(\w+):(\w+):(\w+)', ag.config).groups()

# # find the latest checkpoint
# if ag.resume:
#     checkpointdir = 'logs_' + re.sub(r':', '-', ag.config)
#     path = os.path.join(checkpointdir, 'lightning_logs/version_*')
#     ndir = utils.nsort(glob.glob(path), r'.*version_(\d+)')[0]
#     path = os.path.join(ndir, 'checkpoints/epoch=*')
#     nchk = utils.nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
#     print(f'>> Discovered the latest ckpt {nchk} ..')
#     ag.checkpoint = nchk

# print('>> Initializing Model & Arguments ...')
# model = getattr(models, ag.model).Model(
#     dataset=ag.dataset, loss=ag.loss)

# # experimental features
# if ag.svd:
#     model.do_svd = True

# print('>> Initializing Optimizer ...')
# other_options = {}
# if ag.dp:
#     other_options['accelerator'] = 'dp'
# elif ag.gpus > 1:
#     other_options['accelerator'] = 'ddp'
# else:
#     pass
# if ag.clip > 0.0:
#     other_options['gradient_clip_val'] = ag.clip
# else:
#     pass
# # checkpoint_callback = thl.callbacks.ModelCheckpoint(
# #        monitor=ag.monitor,
# #        mode='max')
# if ag.trail:
#     checkpoint_callback = thl.callbacks.ModelCheckpoint(
#         save_top_k=-1)
#     other_options['checkpoint_callback'] = checkpoint_callback
# trainer = thl.Trainer(
#     max_epochs=model.config.maxepoch,
#     gpus=ag.gpus,
#     log_every_n_steps=1,
#     val_check_interval=1.0,
#     check_val_every_n_epoch=model.config.validate_every,
#     default_root_dir='logs_' + re.sub(r':', '-', ag.config),
#     resume_from_checkpoint=ag.checkpoint if ag.resume else None,
#     **other_options,
# )
# # checkpoint_callback=checkpoint_callback)
# # print(checkpoint_callback.best_model_path)

# print('>> Start Training ...')
# trainer.fit(model)
# if ag.do_test:
#     trainer.test(model)

# print('>> Pulling Down ...')


import sys
sys.path.append('.')
import consle
consle.Train(sys.argv[1:])