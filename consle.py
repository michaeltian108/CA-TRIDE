FAISS_CPU=1
import argparse
import os
import pytorch_lightning as thl
import re
import torch as th
import gc
import psutil
import json
import itertools as it
import numpy as np
import glob
import rich
import torchvision as vision
import configs
import datasets
import models
import utility
import utility.utils as utils
import attacks
import losses
import tensorflow
from models import template_rank
from transformers import AutoFeatureExtractor

from lightning_transformers.task.vision.image_classification import (
    ImageClassificationDataModule,
    ImageClassificationTransformer,
)




c = rich.get_console()


class TFdump:
    '''
    Dump a tensorboard tfevent binary file
    '''

    def __init__(self, *args):
        ag = argparse.ArgumentParser()
        ag.add_argument('-f', '--file', type=str, required=True)
        ag = ag.parse_args(*args)

        from tensorflow.python.summary.summary_iterator import summary_iterator
        for s in summary_iterator(ag.file):
            # print(s)
            if len(s.summary.value) > 0:
                if 'Valid' in s.summary.value[0].tag:
                    c.print('[red]' + str(s.summary.value[0].tag))
                    c.print(f'[yellow]{s.step}', end=' ')
                    c.print('[blue]' + str(s.summary.value[0].simple_value))


class Swipe:
    '''
    Conduct a batch of advrank attack
    it will get stuck if we do not kill the child processes
    invoke script bin/swipe.py to use this cmdline function.
    '''
    profile_eccv28 = (
        #'ES:eps=0.3:alpha=(math 2/255):pgditer=32',
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'),
            ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.1:alpha=0.01:pgditer=20',
                'eps=0.3:alpha=0.01:pgditer=30',
            ))],
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'),
            ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.1:alpha=0.01:pgditer=20',
                'eps=0.3:alpha=0.01:pgditer=30',
            ))],
    )
    profile_eccv224 = (
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'),
            ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=24',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.06:alpha=0.006:pgditer=20',
            ))],
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'),
            ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.06:alpha=0.006:pgditer=20',
            ))],
    )
    profile_pami28 = (
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'), ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.03137:alpha=0.003922:pgditer=32',
                'eps=0.30196:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.03137:alpha=0.003922:pgditer=32',
                'eps=0.30196:alpha=0.011764:pgditer=32',
            ))],
    )
    profile_pami224 = (
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'), ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.00784:alpha=0.003922:pgditer=32',
                'eps=0.03137:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.00784:alpha=0.003922:pgditer=32',
                'eps=0.03137:alpha=0.011764:pgditer=32',
            ))],
    )
    profile_rob28 = (
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=1',), (
                'eps=0.30196:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('QA',), ('pm=+', 'pm=-'), ('M=1',), (
                'eps=0.30196:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('TMA', 'ES', 'LTM', 'GTM', 'GTT'), (
                'eps=0.30196:alpha=0.011764:pgditer=32',
            ))],
    )
    profile_rob224 = (
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=1',), (
                'eps=0.03137:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('QA',), ('pm=+', 'pm=-'), ('M=1',), (
                'eps=0.03137:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('TMA', 'ES', 'LTM', 'GTM', 'GTT'), (
                'eps=0.03137:alpha=0.011764:pgditer=32',
            ))],
    )
    profile_qccurve28 = (
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=1',), (
                f'eps={7*i/255.:.5f}:alpha={max(1,np.round(i*7/25))/255.:.5f}:pgditer=32'
                for i in range(0, 11 + 1))
        )],
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'), ('M=1',), (
                f'eps={7*i/255.:.5f}:alpha={max(1,np.round(i*7/25))/255.:.5f}:pgditer=32'
                for i in range(0, 11 + 1))
        )],
    )

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        g = ag.add_mutually_exclusive_group(required=True)
        g.add_argument('-C', '--checkpoint', type=str, default=None,
                       help='example: logs_mnist-c2f1-ptripletE/lightning_logs/'
                       + 'version_0/checkpoints/epoch=7.ckpt')
        g.add_argument('-c', '--checkpointdir', type=str, default=None,
                       help='example: logs_mnist-c2f1-ptripletE')
        ag.add_argument('-D', '--device', type=str, default='cuda'
                        if th.cuda.is_available() else 'cpu')
        ag.add_argument('-p', '--profile', type=str, required=True,
                        choices=('pami28', 'pami224', 'eccv28', 'eccv224',
                                 'rob28', 'rob224', 'qccurve28'))
        ag.add_argument('-b', '--batchsize', type=int, default=-1)
        ag.add_argument('-m', '--maxiter', type=int, default=None)
        ag.add_argument('-v', '--verbose', action='store_true')
        ag = ag.parse_args(argv)
        profile = getattr(self, 'profile_' + ag.profile)

        # shorthand usage
        if ag.checkpointdir is not None:
            path = os.path.join(ag.checkpointdir, 'lightning_logs/version_*')
            ndir = utility.utils.nsort(glob.glob(path), r'.*version_(\d+)')[0]
            path = os.path.join(ndir, 'checkpoints/epoch=*')
            nchk = utility.utils.nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
            c.print(
                f'[cyan]* Automatically discovered the latest checkpoint .. {nchk}')
            ag.checkpoint = nchk

        instances = {}
        for atk in profile:
            argv = ['-D', ag.device, '-C', ag.checkpoint, '-A', atk]
            if ag.batchsize > 0:
                argv.extend(['-b', str(ag.batchsize)])
            if ag.maxiter is not None:
                argv.extend(['-m', str(ag.maxiter)])
            if ag.verbose:
                argv.append('-v')
            print('Calling AdvRank with', argv)
            instance = AdvRank(argv)
            instances[atk] = instance.stats
            del instance
            children = psutil.Process(os.getpid()).children(recursive=True)
            for child in children:
                child.terminate()
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                p.kill()
            gc.collect()
        results = {key: val for (key, val) in instances.items()}
        c.print(
            '[white on red]=== Final Swipe Results ====================================')
        code = json.dumps(results, indent=2)
        print(utility.utils.rjson(code))
        with open(ag.checkpoint + f'.{ag.profile}.json', 'wt') as f:
            f.write(code)


class AdvClass:
    '''
    Conduct adversarial attack against deep classifier
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--checkpoint', type=str, required=True)
        ag.add_argument('-A', '--attack', type=str, required=True)
        ag.add_argument('-D', '--device', type=str, default='cuda'
                        if th.cuda.is_available() else 'cpu')
        ag.add_argument('-v', '--verbose', action='store_true')
        ag.add_argument('-m', '--maxiter', type=int, default=None)
        ag = ag.parse_args(argv)
        ag.dataset, ag.model, ag.loss = re.match(
            r'.*logs_(\w+)-(\w+)-(\w+)/.*\.ckpt', ag.checkpoint).groups()
        c.print(rich.panel.Panel(' '.join(argv), title='RobRank::AdvClass',
                                 style='bold magenta'))
        c.print(vars(ag))

        c.print('[white on magenta]>_< Restoring Model from Checkpoint ...')
        model = getattr(models, ag.model).Model.load_from_checkpoint(
            checkpoint_path=ag.checkpoint,
            dataset=ag.dataset, loss=ag.loss)
        model = model.to(ag.device)

        c.print('[white on magenta]>_< Initializing Attack Launcher ...')
        atker = attacks.AdvClassLauncher(ag.attack, ag.device, ag.verbose)
        print(atker)

        c.print('[white on magenta]>_< Getting Validation Loader ...')
        model.setup()
        val_dataloader = model.val_dataloader()
        sorig, sadv = atker(model, val_dataloader, maxiter=ag.maxiter)
        self.stats = (sorig, sadv)


class AdvRank:
    '''
    Conduct adversarial attack against ranking (deep metric learning)
    invoke script bin/advrank.py to use this cmdline functionality.
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--checkpoint', type=str, required=True,
                        help='example: logs_mnist-c2f1-ptripletE/lightning_logs/'
                        + 'version_0/checkpoints/epoch=7.ckpt')
        ag.add_argument('-A', '--attack', type=str, required=True)
        ag.add_argument('-D', '--device', type=str, default='cuda'
                        if th.cuda.is_available() else 'cpu')
        ag.add_argument('-v', '--verbose', action='store_true')
        ag.add_argument('-m', '--maxiter', type=int, default=None)
        ag.add_argument('-b', '--batchsize', type=int, default=-1,
                        help='override batchsize')
        ag.add_argument('-X', '--dumpaxd', type=str, default='',
                        help='path to dump the adversarial examples')
        ag = ag.parse_args(argv)
        ag.dataset, ag.model, ag.loss = re.match(
            r'.*logs_(\w+)-(\w+)-(\w+)/.*\.ckpt', ag.checkpoint).groups()
        c.print(rich.panel.Panel(' '.join(argv), title='RobRank::AdvRank',
                                 style='bold magenta'))
        c.print(vars(ag))

        c.print('[white on magenta]>_< Restoring Model from Checkpoint ...')
        model = getattr(models, ag.model).Model.load_from_checkpoint(
            checkpoint_path=ag.checkpoint,
            dataset=ag.dataset, loss=ag.loss)
        model = model.to(ag.device)
        if ag.batchsize > 0:
            model.config.valbatchsize = ag.batchsize

        c.print('[white on magenta]>_< Initializing Attack Launcher ...')
        atker = attacks.AdvRankLauncher(
            ag.attack, ag.device, ag.dumpaxd, ag.verbose)
        print(atker)

        c.print('[white on magenta]>_< Getting Validation Loader ...')
        model.setup()
        val_dataloader = model.val_dataloader()
        sorig, sadv = atker(model, val_dataloader, maxiter=ag.maxiter)
        self.stats = (sorig, sadv)


class Validate:
    '''
    Validate a trained model
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--checkpoint', type=str, default=None)
        ag.add_argument('-g', '--gpus', type=int,
                        default=th.cuda.device_count())
        ag = ag.parse_args(argv)
        ag.dataset, ag.model, ag.loss = re.match(
            r'logs_(\w+)-(\w+)-(\w+)/', ag.checkpoint).groups()
        print(vars(ag))
        model = getattr(models, ag.model).Model(
            dataset=ag.dataset, loss=ag.loss)
        #model.load_from_checkpoint(ag.checkpoint, dataset=ag.dataset, loss=ag.loss)
        trainer = thl.Trainer(gpus=1,#ag.gpus,
                              num_sanity_val_steps=-1,
                              resume_from_checkpoint=ag.checkpoint)
        c.print('[white on magenta]>_< Start Validating ...')
        trainer.validate(model,ckpt_path=ag.checkpoint)
        c.print('[white on red]>_< Pulling Down ...')


class Train:
    '''
    Train a ranking model
    invoke script bin/train.py to use this cmdline functionality.
    '''
    print('Check point for class Train called!')
    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--config', type=str, required=True,
                        help='example: "sop:res18:ptripletE".')
        ag.add_argument('-g', '--gpus', type=int, default=th.cuda.device_count(),
                        help='number of GPUs to use')
        ag.add_argument('--dp', action='store_true',
                        help='use th.nn.DataParallel instead of distributed.')
        ag.add_argument('--do_test', action='store_true')
        ag.add_argument('-m', '--monitor', type=str, default='Validation/r@1')
        ag.add_argument('-r', '--resume', action='store_true')
        ag.add_argument('--clip', type=float, default=0.0,
                        help='do gradient clipping by value')
        ag.add_argument('--trail', action='store_true',
                        help='keep the intermediate checkpoints')
        ag.add_argument('--svd', action='store_true')
        ag = ag.parse_args(argv)
        c.print(rich.panel.Panel(' '.join(argv), title='RobRank::Train',
                                 style='bold magenta'))
        c.print(vars(ag))
        ag.dataset, ag.model, ag.loss = re.match(
            r'(\w+):(\w+):(\w+)', ag.config).groups()

        # find the latest checkpoint
        if ag.resume:
            checkpointdir = 'logs_' + re.sub(r':', '-', ag.config)
            path = os.path.join(checkpointdir, 'lightning_logs/version_*')
            ndir = utility.utils.nsort(glob.glob(path), r'.*version_(\d+)')[0]
            path = os.path.join(ndir, 'checkpoints/epoch=*')
            nchk = utility.utils.nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
            c.print(f'[cyan]* Discovered the latest ckpt .. {nchk}')
            ag.checkpoint = nchk

        c.print('[white on magenta]>_< Initializing Model & Arguments ...')
        model = getattr(models, ag.model).Model(
            dataset=ag.dataset, loss=ag.loss)


                
        # Load model directly
        # from transformers import AutoImageProcessor, AutoModelForImageClassification

        # processor = AutoImageProcessor.from_pretrained(path = "/data1/tqw/rob_IR/models")
        # model = AutoModelForImageClassification.from_pretrained(path = "/data1/tqw/rob_IR/models")
        # experimental features
        if ag.svd:
            model.do_svd = True

        print('>>Initializing Optimizer ...')
        other_options = {}
        # if ag.dp:
        #     other_options['accelerator'] = 'dp'
        # elif ag.gpus > 1:
        #     other_options['accelerator'] = 'ddp'
        # else:
        #     pass
        if ag.clip > 0.0:
            other_options['gradient_clip_val'] = ag.clip
        else:
            pass
        # checkpoint_callback = thl.callbacks.ModelCheckpoint(
        #        monitor=ag.monitor,
        #        mode='max')
        if ag.trail:
            checkpoint_callback = thl.callbacks.ModelCheckpoint(
                save_top_k=-1)
            other_options['checkpoint_callback'] = checkpoint_callback
        trainer = thl.Trainer(
            max_epochs= model.config.maxepoch,# + template_rank.burnin_period,
                        #4*(model.config.maxepoch//3)+20,
                        #model.config.maxepoch,  
                        #20,
            accelerator="gpu", 
            # devices=3,
            # strategy= 'ddp', 
            # gpus=ag.gpus, #argparser给定的显卡数量
            log_every_n_steps=1,
            val_check_interval=1.0,
            check_val_every_n_epoch=model.config.validate_every,
            default_root_dir='logs_' + re.sub(r':', '-', ag.config),
            resume_from_checkpoint=ag.checkpoint if ag.resume else None,
           #strategy=DDPStrategy(find_unused_parameters=False),
           # **other_options,
        )
        # checkpoint_callback=checkpoint_callback
        # print(checkpoint_callback.best_model_path)

        print('>> Start Training ...')
        trainer.fit(model)
        if ag.do_test:
            trainer.test(model)

        print('>> Pulling Down ...')

class Download:
    '''
    Download MNIST and Fashion-MNIST datasets
    '''

    def __init__(self):
        print('>>MNIST')
        vision.datasets.MNIST('~/datasets/', download=True)

        print('>>FashionMNIST')
        vision.datasets.FashionMNIST('~/datasets/', download=True)

        print('>> Done!')