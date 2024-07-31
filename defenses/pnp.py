''' ================================================================================================================================== '''
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
# defenses/pnp.py
# Positive-Negative Perplexing
# Also known as Anti-Collapse Triplet Defense in the paper.
# Some other defense methods, such as REST is also presented here.
###############################################################################

import math
from multiprocessing.dummy import current_process
from typing import Tuple
import re
from sklearn.metrics import dcg_score
import torch as th
import numpy as np
import torch.nn.functional as F
import rich
import sys
sys.path.append('/data1/tqw/rob_IR/')
# from rob_IR.attacks import advrank
import models
from models import template_rank
import datasets
import configs
from utility import utils
from losses.miner import miner
from attacks import AdvRank
import configs
import os
import random
import heapq
import tensorflow as tf
# c = rich.get_console()

global assist_or_not 
global Perturbing_method
Perturbing_method = 'Candidate'

last_modify_epoch = 0.0
class PositiveNegativePerplexing(object):
    '''
    Attack designed for adversarial training
    '''

    def __init__(self,
                 model: th.nn.Module, eps: float, alpha: float, pgditer: int,
                 device: str, metric: str, verbose: bool = False):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.pgditer = pgditer
        self.device = device
        self.metric = metric
        self.verbose = verbose

   
    def CAP(self, images: th.Tensor, triplets: tuple,  model_name, dataset_name, loss_name,epoch,maxepoch, bs):
        '''
        Helping DNN to separate N and P 
        '''
        global last_modify_epoch
        global last_c
        # global lbd
        lbd_p = 10.
        lbd_n = lbd_p
        anc, pos, neg = triplets
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        imanc = images[anc, :, :, :]#.clone().detach().to(self.device)
        images_orig = th.cat([impos,imneg]).clone().detach()
        images = th.cat([impos,imneg])
        # images_vul = th.cat([impos,imneg])
        images.requires_grad = True
        adv_decay =   1.*epoch/maxepoch
        # start PGD
        self.model.eval()

        for iteration in range(16):
            # optimizer
            optm = th.optim.SGD(self.model.parameters(), lr=0.)
            optx = th.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            emb_anc = self.model.forward(imanc)
            emb_orig = self.model.forward(images_orig)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
                emb_anc = F.normalize(emb_anc)
                emb_orig = F.normalize(emb_orig)
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                dis_ap = F.pairwise_distance(emb[:len(emb) // 2],
                                           emb_anc)#.mean()
                dis_an = F.pairwise_distance(emb_anc,emb[len(emb) // 2:])#.mean()
                dif_np =  dis_ap.mean() - dis_an.mean()
                '''
                Calculate Collapsness
                '''
                w_ap = th.exp( lbd_p*(dis_ap - max(dis_ap)))
                w_an = th.exp( lbd_n*(dis_an - max(dis_an)))
                w_m_ap = (w_ap*dis_ap).sum()/(w_ap.sum())
                w_m_an = (w_an*dis_an).sum()/(w_an.sum())
          
                if iteration == 0 :
                    h = dis_ap.mean() - dis_an.mean()
                    c = w_m_ap - w_m_an
                    current_c = c.clone().detach()

                loss = (w_m_an - w_m_ap).relu()

            if loss <1e-5:
                break
            
            itermsg = {'loss': loss.item()}
            loss.backward()
            if self.pgditer > 1 :
                images.grad.data.copy_(self.alpha*th.sign(images.grad)*adv_decay)
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad)*adv_decay)
            optx.step()
  
                    
            images = th.min(images, images_orig + self.eps )
            images = th.max(images, images_orig - self.eps )
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            if self.verbose:
                print(f'(PGD)>', itermsg)
        

        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        #clean the distances
        dis_an = 0.0
        dis_ap = 0.0
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        # images: concatenation of adversarial positive and negative
        return images
        
    def ANP(self, images: th.Tensor, triplets: tuple,  model_name, dataset_name, loss_name,epoch,maxepoch): #, bs)
        '''
        Only perturb anchor point to initiate real world R@1 targted attack
        '''
        anc, pos, neg = triplets
        impos = images[pos, :, :, :]#.clone().detach().to(self.device)
        imneg = images[neg, :, :, :]#.clone().detach().to(self.device)
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        images_orig =imanc.clone().detach()
        images = imanc
        images.requires_grad = True
        adv_decay = 1.*epoch/maxepoch
        # start PGD
        lbd_p = 10.
        lbd_n = lbd_p
        self.model.eval()
        pgd_qa_n = 16

        for iteration in range(pgd_qa_n):
            # optimizer
            optm = th.optim.SGD(self.model.parameters(), lr=0.)
            optx = th.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb_orig_anc = self.model.forward(images_orig)
            emb_anc = self.model.forward(images)
            emb_pos = self.model.forward(impos)
            emb_neg = self.model.forward(imneg)
            if self.metric in ('C', 'N'):
                emb_anc = F.normalize(emb_anc)
                emb_pos = F.normalize(emb_pos)
                emb_neg = F.normalize(emb_neg)
                emb_orig_anc = F.normalize(emb_orig_anc)
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                dis_ap = F.pairwise_distance(emb_pos,
                                           emb_anc)
                dis_an = F.pairwise_distance(emb_anc,emb_neg)
                dis_np = F.pairwise_distance(emb_neg, emb_pos).mean()
                dif_np = dis_ap.mean() - dis_an.mean()
                top_len = len(dis_an)//2
                _, an_top_half_idx = th.topk(dis_an, top_len, largest = False)
                _, ap_top_half_idx = th.topk(dis_ap, top_len, largest = False)
                _, an_last_half_idx = th.topk(dis_an, top_len, largest = True)
                an_top_half = dis_an[an_top_half_idx].mean()
                dis_aa = F.pairwise_distance(emb_anc, emb_orig_anc).mean()

                '''
                Calculate Collapseness
                '''
                w_ap = th.exp( lbd_p*(dis_ap - max(dis_ap)))
                w_an = th.exp( lbd_n*(dis_an - max(dis_an)))
                w_m_ap = (w_ap*dis_ap).sum()/(w_ap.sum())
                w_m_an = (w_an*dis_an).sum()/(w_an.sum())

                if iteration == 0:
                    c = w_m_ap - w_m_an 
                    h = dis_ap.mean() - dis_an.mean()

                gama = math.exp( - (w_m_an - w_m_ap).relu())
                loss = (w_m_ap - w_m_an ).relu() + gama * ( an_top_half - dis_aa).relu()
            if loss <1e-5:
                break

            itermsg = {'loss': loss.item()}
            loss.backward()
            if self.pgditer > 1 :
                images.grad.data.copy_(self.alpha*th.sign(images.grad)*adv_decay)
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad)*adv_decay)
            optx.step()
            images = th.min(images, images_orig + self.eps)
            images = th.max(images, images_orig - self.eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        #clean the distances
        dis_an = 0.0
        dis_ap = 0.0
        w_m_an = 0.0
        w_m_ap = 0.0
        an_top_half = 0.0
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        # images: concatenation of adversarial positive and negative
        return images

    
def pnp_training_step(model: th.nn.Module, batch, batch_idx, model_name, dataset_name, epoch, maxepoch,batch_size, *,
                      pgditer: int = None):
    global Perturbing_method
    '''
    Adversarial training with Positive/Negative Perplexing (PNP) Attack.
    Function signature follows pytorch_lightning.LightningModule.training_step,
    where model is a lightning model, batch is a tuple consisting images
    (th.Tensor) and labels (th.Tensor), and batch_idx is just an integer.

    Collapsing positive and negative -- Anti-Collapse (ACO) defense.
    force the model to learning robust feature and prevent the
    adversary from exploiting the non-robust feature and collapsing
    the positive/negative samples again. This is exactly the ACT defense
    discussed in https://arxiv.org/abs/2106.03614

    This defense is not agnostic to backbone architecure and metric learning
    loss. But it is recommended to use it in conjunction with triplet loss.
    '''
    # check loss function
    
    if not re.match(r'p.?triplet.*', model.loss) and \
            not re.match(r'psnr.*', model.loss) and \
            not re.match(r'pmargin.*', model.loss) and \
            not re.match(r'pcontrast.*', model.loss) and \
            not re.match(r'pgcontrast.*', model.loss) and \
            not re.match(r'pLDAtrip.*', model.loss):
        raise ValueError(f'ACT defense is not implemented for {model.loss}!')

    # prepare data batch in a proper shape
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
    elif model.dataset in ('mnist', 'fashion'):
        images = images.view(-1, 1, 28, 28)
    else:
        raise ValueError(f'possibly illegal dataset {model.dataset}?')
    # evaluate original benign sample
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, epoch, maxepoch, Perturbing_method, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=model.config.advtrain_eps,
                                     alpha=model.config.advtrain_alpha,
                                     pgditer=model.config.advtrain_pgditer if pgditer is None else pgditer,
                                     device=model.device, metric=model.metric,
                                     verbose=False)
    # Collapsing positive and negative -- Anti-Collapse Triplet (ACT) defense.
    model.eval()
    model.wantsgrad = True
    if hasattr(model, 'is_advtrain_pnp_adapt') and model.is_advtrain_pnp_adapt:
        if re.match(r'ptriplet.*', model.loss):
            #print('>>Engaging trip_chase AT...')
            if not assist_or_not:
                images_pnp = pnp.act_triplet_chase(images, triplets,model_name,dataset_name,model.loss)
            else:
                images_pnp = pnp.assist(images, triplets,model_name,dataset_name,model.loss)
        elif re.match(r'pLDAtrip.*', model.loss):
            images_pnp = pnp.triplet_chase(images, triplets,model_name,dataset_name,model.loss)
        # adapt pnp/act for the specific loss function
        elif re.match(r'pcontrast.*', model.loss):
            assert(model.loss == 'pcontrastN')
            with th.no_grad():
                mask = F.pairwise_distance(
                    output_orig[anc, :], output_orig[neg, :]) < configs.contrastive.margin_euclidean
                mask = mask.view(-1, 1)
            images_pnp = pnp.pncollapse(images, triplets)
            images_aps = pnp.apsplit(images, triplets)

            model.train()
            epnp = F.normalize(model.forward(images_pnp))
            eaps = F.normalize(model.forward(images_aps))
            model.wantsgrad = False
            N = len(anc)
            ep = th.where(mask, epnp[:N], eaps[:N])
            en = th.where(mask, epnp[N:], eaps[N:])
            model.train()
            loss = model.lossfunc.raw(output_orig[anc, :], ep, en).mean()
            
            
            #
            model.log('Train/loss_orig', loss_orig.item())
            model.log('Train/loss_adv', loss.item())
            return loss
        else:
            raise NotImplementedError(
                f'not implemeneted pnp/act for {model.loss}')
    else:
            if Perturbing_method == 'Query' :#epoch >= maxepoch -1:
                images_pnp = pnp.ANP(images, triplets,model_name,dataset_name,model.loss,epoch,maxepoch)
                Perturbing_method = 'Candidate'
            
            #     Passing images to model and start adversarial training
                model.train()
                aemb = model.forward(images_pnp)
                pemb = model.forward(images[pos, :, :, :])
                nemb = model.forward(images[neg, :, :, :])
            else: 
                images_pnp = pnp.CAP(images, triplets,model_name,dataset_name,model.loss,epoch,maxepoch, batch_size)
                Perturbing_method ='Query'

            #     Passing images to model and start adversarial training
                model.train()
                pnemb = model.forward(images_pnp)
                aemb = model.forward(images[anc, :, :, :])#th.cat((aemb, model.forward(images[anc, :, :, :])))
                pemb = pnemb[:len(pnemb) // 2]#th.cat((pemb,pnemb[:len(pnemb) // 2]))
                nemb = pnemb[len(pnemb) // 2:]#th.cat((nemb,pnemb[len(pnemb) // 2:]))
    # Adversarial Training
    if model.lossfunc._metric in ('C', 'N'):
        pemb = F.normalize(pemb)
        aemb = F.normalize(aemb)
        nemb = F.normalize(nemb)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(aemb,pemb,
                              nemb, epoch, maxepoch).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss

