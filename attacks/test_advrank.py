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

# pylint: disable=no-member,invalid-envvar-default,unused-argument,attribute-defined-outside-init
import os
import json
import torch as th
from tqdm import tqdm
import statistics
import numpy as np
import torch.nn.functional as F
import functools as ft
try:
    from .advrank_qcselector import QCSelector
    from .advrank_loss import AdvRankLoss
except ImportError:
    from advrank_qcselector import QCSelector
    from advrank_loss import AdvRankLoss
import pytest
from termcolor import colored
import itertools as it
import sys
sys.path.append('/home/tianqiwei/jupyter/rob_IR/')
import datasets
import configs
from utility import utils
from utils import rjson


class AdvRank(object):
    '''
    Overall implementation of Adversarial Ranking Attack
    '''

    def __init__(self, model: th.nn.Module, *,
                 eps: float = 4. / 255., alpha: float = 1. / 255., pgditer: int = 24,
                 attack_type: str = None,
                 M: int = None, W: int = None, pm: str = None,
                 verbose: bool = False, device: str = 'cpu',
                 metric: str = None):

        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.pgditer = pgditer
        self.attack_type = attack_type
        self.M = M
        self.W = W
        self.pm = pm
        self.verbose = verbose
        self.device = device
        self.metric = metric
        self.XI = 1.

    def __call__(self, images, labels, candi) -> tuple:
        '''
        Main entrance of this class
        '''
        return self.attack(images, labels, candi)

    def update_xi(self, loss_sp):
        '''
        ECCV20 paper (2002.11293) uses a fixed xi parameter.
        Here we use a dynamic one which does not participate in backprop.
        '''
        if not hasattr(self.model, 'dataset'):
            # we are running pytest.
            self.XI = 1e0
            return
        if isinstance(loss_sp, th.Tensor):
            loss_sp = loss_sp.item()
        # self.XI = np.exp(loss_sp.item() * 5e4) # very strict
        if any(x in self.model.dataset for x in ('mnist',)):
            self.XI = np.min((np.exp(loss_sp * 2e4), 1e9))
        elif any(x in self.model.dataset for x in ('fashion',)):
            self.XI = np.min((np.exp(loss_sp * 4e4), 1e9))
        elif any(x in self.model.dataset for x in ('cub',)):
            self.XI = np.min((np.exp(loss_sp * 4e4), 1e9))
        elif any(x in self.model.dataset for x in ('cars',)):
            self.XI = np.min((np.exp(loss_sp * 7e4), 1e9))
        elif any(x in self.model.dataset for x in ('sop',)):
            self.XI = np.min((np.exp(loss_sp * 7e4), 1e9))
        else:
            raise NotImplementedError

    def forwardmetric(self, images: th.Tensor) -> th.Tensor:
        '''
        metric-aware forwarding
        '''
        output = self.model.forward(images)
        if self.metric in ('C', 'N'):
            return F.normalize(output)
        elif self.metric in ('E', ):
            return output

    def outputdist(self, images: th.Tensor, labels: th.Tensor,
                   candi: tuple) -> tuple:
        '''
        calculate output, and dist w.r.t. candidates.

        Note, this function does not differentiate. It's used for evaluation
        '''
        self.model.eval()
        with th.no_grad():
            if self.metric == 'C':
                output = self.forwardmetric(images)
                # [num_output_num, num_candidate]
                dist = 1 - output @ candi[0].t()
            elif self.metric in ('E', 'N'):
                output = self.forwardmetric(images)
                dist = th.cdist(output, candi[0])
                # the memory requirement is insane
                # should use more efficient method for Euclidean distance
                # dist2 = []
                # for i in range(output.shape[0]):
                #    xq = output[i].view(1, -1)
                #    xqd = (candi[0] - xq).norm(2, dim=1).squeeze()
                #    dist2.append(xqd)
                # dist2 = th.stack(dist2)  # [num_output_num, num_candidate]
                #assert((dist2 - dist).norm() < 1e-3)
            else:
                raise ValueError(self.metric)
            output_detach = output.clone().detach()
            dist_detach = dist.clone().detach()
        return (output_detach, dist_detach)

    def eval_advrank(self, images, labels, candi, *, resample=True) -> dict:
        '''
        evaluate original images or adversarial images for ranking
        `resample` is used for retaining selection for multiple times of evals.

        side effect:
            it sets self.qcsel when resample is toggled
        '''
        # evaluate original output and dist
        output, dist = self.outputdist(images, labels, candi)
        attack_type = self.attack_type
        M, W = self.M, self.W

        # [[[ dispatch: qcselection and original evaluation ]]]
        # -> dispatch: ES
        if (attack_type == 'ES'):
            # select queries and candidates for ES
            if resample:
                self.qcsel = QCSelector('ES', M, W, False)(dist, candi)
                self.output_orig = output.clone().detach()
            output_orig = self.output_orig
            # evaluate the attack
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1, r_10, r_100 = [], [], []
            if resample:
                for i in range(dist.shape[0]):
                    agsort = dist[i].cpu().numpy().argsort()[1:]
                    rank = np.where(allLab[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
            else:
                # We are now evaluating adversarial examples
                # hence masking the query itself in this way
                for i in range(dist.shape[0]):
                    if self.metric == 'C':
                        loc = 1 - candi[0] @ output_orig[i].view(-1, 1)
                        loc = loc.flatten().argmin().cpu().numpy()
                    else:
                        loc = (candi[0] - output_orig[i]).norm(2, dim=1)
                        loc = loc.flatten().argmin().cpu().numpy()
                    dist[i][loc] = 1e38  # according to float32 range.
                    agsort = dist[i].cpu().numpy().argsort()[0:]
                    rank = np.where(allLab[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
            r_1, r_10, r_100 = 100 * \
                np.mean(r_1), 100 * np.mean(r_10), 100 * np.mean(r_100)
            loss, _ = AdvRankLoss('ES', self.metric)(output, output_orig)
            # summary
            summary_orig = {'loss': loss.item(), 'r@1': r_1,
                            'r@10': r_10, 'r@100': r_100}

        # -> dispatch: LTM
        elif attack_type == 'LTM':
            if resample:
                self.output_orig = output.clone().detach()
                self.loc_self = dist.argmin(dim=-1).view(-1)
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1 = []
            for i in range(dist.size(0)):
                dist[i][self.loc_self[i]] = 1e38
                argsort = dist[i].cpu().numpy().argsort()[0:]
                rank = np.where(allLab[argsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
            r_1 = np.mean(r_1)
            # summary
            summary_orig = {'r@1': r_1}

        # -> dispatch: TMA
        elif attack_type == 'TMA':
            if resample:
                self.output_orig = output.clone().detach()
                self.qcsel = QCSelector('TMA', None, None)(dist, candi)
            (embrand, _) = self.qcsel
            cossim = F.cosine_similarity(output, embrand).mean().item()
            # summary
            summary_orig = {'Cosine-SIM': cossim}

        # -> dispatch: GTM
        elif (attack_type == 'GTM'):
            if resample:
                self.output_orig = output.clone().detach()
                self.dist_orig = dist.clone().detach()
                self.loc_self = self.dist_orig.argmin(dim=-1).view(-1)
                self.qcsel = QCSelector('GTM', None, None)(dist, candi,
                                                           self.loc_self)
            output_orig = self.output_orig
            # evaluate the attack
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1 = []
            # the process is similar to that for ES attack
            # except that we only evaluate recall at 1 (r_1)
            for i in range(dist.shape[0]):
                dist[i][self.loc_self[i]] = 1e38
                argsort = dist[i].cpu().numpy().argsort()[0:]
                rank = np.where(allLab[argsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
            r_1 = np.mean(r_1)
            # summary
            summary_orig = {'r@1': r_1}

        # -> dispatch: GTT
        elif (attack_type == 'GTT'):
            if resample:
                self.output_orig = output.clone().detach()
                self.dist_orig = dist.clone().detach()
                self.loc_self = self.dist_orig.argmin(dim=-1).view(-1)
                self.qcsel = QCSelector('GTT', None, None)(
                    dist, candi, self.loc_self)
            dist[range(len(self.loc_self)), self.loc_self] = 1e38
            ((_, idm), (_, _), (_, _)) = self.qcsel
            re1 = (dist.argmin(dim=-1).view(-1) == idm).float().mean().item()
            dk = {}
            for k in (4,):
                topk = dist.topk(k, dim=-1, largest=False)[1]
                seq = [topk[:, j].view(-1) == idm for j in range(k)]
                idrecall = ft.reduce(th.logical_or, seq)
                dk[f'retain@{k}'] = idrecall.float().mean().item()
            # summary
            summary_orig = {'ID-Retain@1': re1, **dk}

        # -> dispatch: FOA M=2
        elif (attack_type == 'FOA') and (M == 2):
            # select quries and candidates for FOA(M=2)
            if resample:
                self.qcsel = QCSelector('FOA', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            # >> compute the (ordinary) loss on selected targets
            loss, acc = AdvRankLoss('FOA2', self.metric)(
                output, embpairs[:, 1, :], embpairs[:, 0, :])
            # summary
            summary_orig = {'loss': loss.item(), 'FOA2:Accuracy': acc}

        # -> dispatch: SPFOA M=2
        elif (attack_type == 'SPFOA') and (M == 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss, acc = AdvRankLoss('FOA2', self.metric)(
                output, embpairs[:, 1, :], embpairs[:, 0, :])
            loss_sp, rank_gt = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            self.update_xi(loss_sp)
            loss = loss + self.XI * loss_sp
            # summary
            summary_orig = {'loss': loss.item(), 'loss_sp': loss_sp.item(),
                            'FOA2:Accuracy': acc, 'GT.mR': rank_gt / candi[0].size(0)}

        # -> dispatch: FOA M>2
        elif (attack_type == 'FOA') and (M > 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            loss, tau = AdvRankLoss('FOAX', self.metric)(output, embpairs)
            summary_orig = {'loss': loss.item(), 'FOA:tau': tau}

        # -> dispatch: SPFOA M>2
        elif (attack_type == 'SPFOA') and (M > 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss, tau = AdvRankLoss('FOAX', self.metric)(output, embpairs)
            loss_sp, rank_sp = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            loss = loss + self.XI * loss_sp
            summary_orig = {'loss': loss.item(), 'loss_sp': loss_sp.item(),
                            'FOA:tau': tau, 'GT.mR': rank_sp / candi[0].size(0)}

        # -> dispatch: CA
        elif (attack_type == 'CA'):
            if resample:
                self.qcsel = QCSelector(f'CA{self.pm}', M, W)(dist, candi)
            embpairs, msamples = self.qcsel
            loss, rank = AdvRankLoss(f'CA{self.pm}', self.metric)(
                output, embpairs, candi[0])
            mrank = rank / candi[0].shape[0]
            summary_orig = {'loss': loss.item(), f'CA{self.pm}:prank': mrank}

        # -> dispatch: QA
        elif (attack_type == 'QA'):
            if resample:
                self.qcsel = QCSelector(f'QA{self.pm}', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            loss, rank_qa = AdvRankLoss(f'QA{self.pm}', self.metric)(
                output, embpairs, candi[0], dist=dist, cidx=msample)
            mrank = rank_qa / candi[0].shape[0]  # percentile ranking
            summary_orig = {'loss': loss.item(), f'QA{self.pm}:prank': mrank}

        # -> dispatch: SPQA
        elif (attack_type == 'SPQA'):
            if resample:
                self.qcsel = QCSelector(
                    f'QA{self.pm}', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss_qa, rank_qa = AdvRankLoss(f'QA{self.pm}', self.metric)(
                output, embpairs, candi[0], dist=dist, cidx=msample)
            loss_sp, rank_sp = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            self.update_xi(loss_sp)
            loss = loss_qa + self.XI * loss_sp
            mrank = rank_qa / candi[0].shape[0]
            mrankgt = rank_sp / candi[0].shape[0]
            summary_orig = {'loss': loss.item(), f'SPQA{self.pm}:prank': mrank,
                            f'SPQA{self.pm}:GTprank': mrankgt}

        # -> dispatch: N/A
        else:
            raise Exception("Unknown attack")
        # note: QCSelector results are stored in self.qcsel
        return output, dist, summary_orig

    def embShift(self, images: th.Tensor, orig: th.Tensor = None) -> th.Tensor:
        '''
        barely performs the ES attack without any evaluation.
        used for adv training. [2002.11293]

        Returns the adversarial example.
        '''
        assert(isinstance(images, th.Tensor))
        images = images.clone().detach().to(self.device)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # evaluate original samples, and set self.qcsel
        with th.no_grad():
            output_orig = orig if orig is not None else self.forwardmetric(
                images)
        # -> start PGD optimization
        self.model.eval()
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = th.optim.SGD(self.model.parameters(), lr=0.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            output = self.forwardmetric(images)

            # calculate differentiable loss
            if iteration == 0:
                noise = 1e-7 * th.randint_like(
                    output_orig, -1, 2, device=output_orig.device)
                # avoid zero gradient
                loss, _ = AdvRankLoss('ES', self.metric)(
                    output, output_orig + noise)
            else:
                loss, _ = AdvRankLoss('ES', self.metric)(
                    output, output_orig)
            itermsg = {'loss': loss.item()}
            loss.backward()

            # >> PGD: project SGD optimized result back to a valid region
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * th.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad))  # FGSM
            optimx.step()
            # L_infty constraint
            images = th.min(images, images_orig + self.eps)
            # L_infty constraint
            images = th.max(images, images_orig - self.eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True

            # itermsg
            if int(os.getenv('PGD', -1)) > 0:
                print('(PGD)>', itermsg)

        # note: It's very critical to clear the junk gradients
        optim.zero_grad()
        optimx.zero_grad()
        images.requires_grad = False

        # evaluate adversarial samples
        xr = images.clone().detach()
        if self.verbose:
            r = images - images_orig
            with th.no_grad():
                output = self.forwardmetric(images)
                # also calculate embedding shift
                if self.metric == 'C':
                    embshift = (1 - F.cosine_similarity(output, output_orig)
                                ).mean().item()
                elif self.metric in ('E', 'N'):
                    embshift = F.pairwise_distance(output, output_orig
                                                   ).mean().item()
            tqdm.write(colored(' '.join(['r>',
                                         f'[{self.alpha:.3f}*{self.pgditer:d}^{self.eps:.3f}]',
                                         'Min', '%.3f' % r.min().item(),
                                         'Max', '%.3f' % r.max().item(),
                                         'Mean', '%.3f' % r.mean().item(),
                                         'L0', '%.3f' % r.norm(
                                             0, 1).mean().item(),
                                         'L1', '%.3f' % r.norm(
                                             1, 1).mean().item(),
                                         'L2', '%.3f' % r.norm(
                                             2, 1).mean().item(),
                                         'embShift', '%.3f' % embshift,
                                         ]),
                               'blue'))
        return xr

    def attack(self, images: th.Tensor, labels: th.Tensor,
               candi: tuple) -> tuple:
        '''
        Note, all images must lie in [0,1]^D
        '''
        # prepare the current batch of data
        assert(isinstance(images, th.Tensor))
        images = images.clone().detach().to(self.device)
        images_orig = images.clone().detach()
        images.requires_grad = True
        labels = labels.to(self.device).view(-1)
        attack_type = self.attack_type

        # evaluate original samples, and set self.qcsel
        with th.no_grad():
            output_orig, dist_orig, summary_orig = self.eval_advrank(
                images, labels, candi, resample=True)
        if self.verbose:
            tqdm.write(colored('* OriEval', 'green', None, ['bold']), end=' ')
            tqdm.write(rjson(json.dumps(summary_orig)), end='')

        # -> with/without random init?
        if int(os.getenv('RINIT', 0)) > 0:
            images = images + self.eps * 2 * \
                (0.5 - th.rand(images.shape)).to(images.device)
            images = th.clamp(images, min=0., max=1.)
            images = images.detach()
            images.requires_grad = True

        # -> start PGD optimization
        # truning the model into tranining mode triggers obscure problem:
        # incorrect validation performance due to BatchNorm. We do automatic
        # differentiation in evaluation mode instead.
        self.model.eval()
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = th.optim.SGD(self.model.parameters(), lr=1.)
            optimx = th.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            output = self.forwardmetric(images)

            # calculate differentiable loss
            if (attack_type == 'ES'):
                if iteration == 0:
                    # avoid zero gradient
                    loss, _ = AdvRankLoss('ES', self.metric)(
                        output, output_orig + 1e-7)
                else:
                    loss, _ = AdvRankLoss('ES', self.metric)(
                        output, output_orig)
                itermsg = {'loss': loss.item()}
            elif (attack_type == 'FOA') and (self.M == 2):
                # >> reverse the inequalities (ordinary: d1 < d2, adversary: d1 > d2)
                embpairs, _ = self.qcsel
                loss, _ = AdvRankLoss('FOA2', self.metric)(
                    output, embpairs[:, 1, :], embpairs[:, 0, :])
                itermsg = {'loss': loss.item()}
            elif (attack_type == 'SPFOA') and (self.M == 2):
                embpairs, _, embgts, _ = self.qcsel
                loss, _ = AdvRankLoss('FOA2', self.metric)(
                    output, embpairs[:, 1, :], embpairs[:, 0, :])
                loss_sp, _ = AdvRankLoss(f'QA{self.pm}', self.metric)(
                    output, embgts, candi[0])
                loss = loss + self. XI * loss_sp
                itermsg = {'loss': loss.item(), 'SP.QA+': loss_sp.item()}
            elif (attack_type == 'FOA') and (self.M > 2):
                # >> enforce the random inequality set (di < dj for all i,j where i<j)
                embpairs, _ = self.qcsel
                loss, _ = AdvRankLoss('FOAX', self.metric)(output, embpairs)
                itermsg = {'loss': loss.item()}
            elif (attack_type == 'SPFOA') and (self.M > 2):
                embpairs, _, embgts, _ = self.qcsel
                loss, _ = AdvRankLoss('FOAX', self.metric)(output, embpairs)
                loss_sp, _ = AdvRankLoss(f'QA{self.pm}', self.metric)(
                    output, embgts, candi[0])
                self.update_xi(loss_sp)
                loss = loss + self.XI * loss_sp
                itermsg = {'loss': loss.item(), 'SP.QA+': loss_sp.item()}
            elif (attack_type == 'CA'):
                embpairs, _ = self.qcsel
                if int(os.getenv('DISTANCE', 0)) > 0:
                    loss, _ = AdvRankLoss(
                        'CA-DIST', self.metric)(output, embpairs, candi[0], pm=self.pm)
                else:
                    loss, _ = AdvRankLoss(f'CA{self.pm}', self.metric)(
                        output, embpairs, candi[0])
                itermsg = {'loss': loss.item()}
            elif (attack_type == 'QA'):
                embpairs, _ = self.qcsel
                # == enforce the target set of inequalities, while preserving the semantic
                if int(os.getenv('DISTANCE', 0)) > 0:
                    loss, _ = AdvRankLoss(
                        'QA-DIST', self.metric)(output, embpairs, candi[0], pm=self.pm)
                else:
                    loss, _ = AdvRankLoss('QA', self.metric)(
                        output, embpairs, candi[0], pm=self.pm)
                itermsg = {'loss': loss.item()}
            elif (attack_type == 'SPQA') and int(os.getenv('PGD', -1)) > 0:
                # THIS IS INTENDED FOR DEBUGGING AND DEVELOPING XI SCHEME
                print('DEBUGGING MODE')
                embpairs, cidx, embgts, gidx = self.qcsel
                with th.no_grad():
                    dist = th.cdist(output, candi[0])
                loss_qa, crank = AdvRankLoss('QA', self.metric)(
                    output, embpairs, candi[0], pm=self.pm,
                    cidx=cidx, dist=dist)
                loss_sp, grank = AdvRankLoss('QA', self.metric)(
                    output, embgts, candi[0], pm='+',
                    cidx=gidx, dist=dist)
                self.update_xi(loss_sp)
                loss = loss_qa + self.XI * loss_sp
                itermsg = {'loss': loss.item(), 'loss_qa': loss_qa.item(),
                           'loss_sp': loss_sp.item(), 'xi': self.XI,
                           'crank': crank / candi[0].shape[0],
                           'grank': grank / candi[0].shape[0]}
            elif (attack_type == 'SPQA'):
                embpairs, _, embgts, _ = self.qcsel
                if int(os.getenv('DISTANCE', 0)) > 0:
                    #raise NotImplementedError("SP for distance based objective makes no sense here")
                    loss_qa, _ = AdvRankLoss(
                        'QA-DIST', self.metric)(output, embpairs, candi[0], pm=self.pm)
                    loss_sp, _ = AdvRankLoss(
                        'QA-DIST', self.metric)(output, embgts, candi[0], pm='+')
                    self.update_xi(loss_sp)
                    loss = loss_qa + self.XI * loss_sp
                else:
                    loss_qa, _ = AdvRankLoss('QA', self.metric)(
                        output, embpairs, candi[0], pm=self.pm)
                    loss_sp, _ = AdvRankLoss('QA', self.metric)(
                        output, embgts, candi[0], pm='+')
                    self.update_xi(loss_sp)
                    loss = loss_qa + self.XI * loss_sp
                itermsg = {'loss': loss.item(), 'loss_qa': loss_qa.item(),
                           'loss_sp': loss_sp.item()}
            elif (attack_type == 'GTM'):
                ((emm, _), (emu, _), (ems, _)) = self.qcsel
                loss = AdvRankLoss('GTM', self.metric)(
                    output, emm, emu, ems, candi[0])
                itermsg = {'loss': loss.item()}
                # Note: greedy qc selection / resample harms performance
                # with th.no_grad():
                #    if self.metric in ('C',):
                #        dist = 1 - output @ candi[0].t()
                #    elif self.metric in ('E', 'N'):
                #        dist = th.cdist(output, candi[0])
                # self.qcsel = QCSelector('GTM', None, None)(dist, candi,
                #        self.dist_orig)
            elif (attack_type == 'GTT'):
                ((emm, idm), (emu, idum), (ems, _)) = self.qcsel
                loss = AdvRankLoss('GTT', self.metric)(
                    output, emm, emu, ems, candi[0])
                itermsg = {'loss': loss.item()}
            elif attack_type == 'TMA':
                (embrand, _) = self.qcsel
                loss = AdvRankLoss('TMA', self.metric)(output, embrand)
                itermsg = {'loss': loss.item()}
            elif attack_type == 'LTM':
                mask_same = (candi[1].view(1, -1) == labels.view(-1, 1))
                mask_same.scatter(1, self.loc_self.view(-1, 1), False)
                mask_diff = (candi[1].view(1, -1) != labels.view(-1, 1))
                if self.metric in ('E', 'N'):
                    dist = th.cdist(output, candi[0])
                elif self.metric == 'C':
                    dist = 1 - output @ candi[0].t()
                maxdan = th.stack([dist[i, mask_diff[i]].max()
                                   for i in range(dist.size(0))])
                mindap = th.stack([dist[i, mask_same[i]].min()
                                   for i in range(dist.size(0))])
                loss = (maxdan - mindap).relu().sum()
                itermsg = {'loss': loss.item()}
            else:
                raise Exception("Unknown attack")
            if self.verbose and int(os.getenv('PGD', -1)) > 0:
                tqdm.write(colored('(PGD)>\t' + json.dumps(itermsg), 'yellow'))
            loss.backward()

            # >> PGD: project SGD optimized result back to a valid region
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * th.sign(images.grad))
                # note: don't know whether this trick helps
                #alpha = self.alpha if iteration %2 == 0 else 2. * self.alpha
                #images.grad.data.copy_(alpha * th.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad))  # FGSM
            optimx.step()
            # L_infty constraint
            images = th.min(images, images_orig + self.eps)
            # L_infty constraint
            images = th.max(images, images_orig - self.eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True

        optim.zero_grad()
        optimx.zero_grad()
        images.requires_grad = False

        # evaluate adversarial samples
        xr = images.clone().detach()
        r = images - images_orig
        if self.verbose:
            tqdm.write(colored(' '.join(['r>',
                                         'Min', '%.3f' % r.min().item(),
                                         'Max', '%.3f' % r.max().item(),
                                         'Mean', '%.3f' % r.mean().item(),
                                         'L0', '%.3f' % r.norm(0).item(),
                                         'L1', '%.3f' % r.norm(1).item(),
                                         'L2', '%.3f' % r.norm(2).item()]),
                               'blue'))
        self.model.eval()
        with th.no_grad():
            output_adv, dist_adv, summary_adv = self.eval_advrank(
                xr, labels, candi, resample=False)

            # also calculate embedding shift
            if self.metric == 'C':
                distance = 1 - th.mm(output, output_orig.t())
                # i.e. trace = diag.sum
                embshift = distance.trace() / output.shape[0]
                summary_adv['embshift'] = embshift.item()
            elif self.metric in ('E', 'N'):
                distance = th.nn.functional.pairwise_distance(
                    output, output_orig, p=2)
                embshift = distance.sum() / output.shape[0]
                summary_adv['embshift'] = embshift.item()

        if self.verbose:
            tqdm.write(colored('* AdvEval', 'red', None, ['bold']), end=' ')
            tqdm.write(rjson(json.dumps(summary_adv)), end='')
        return (xr, r, summary_orig, summary_adv)


@pytest.mark.parametrize('metric, pgditer, eps',
                         it.product(('C', 'E', 'N'), (1, 4), (0., 0.1)))
def test_es(metric, pgditer, eps):
    # pylint: disable=unused-variable
    model = th.nn.Sequential(th.nn.Linear(8, 8))
    print(model)
    dataset = (th.rand(1000, 8), th.randint(0, 10, (1000,)))
    with th.no_grad():
        if metric in ('C', 'N'):
            candi = (F.normalize(model(dataset[0])), dataset[1])
        else:
            candi = (model(dataset[0]), dataset[1])
    print(candi)
    # select first 8 to perturb
    images, labels = dataset[0][:8], dataset[1][:8]

    advrank = AdvRank(model, attack_type='ES', eps=eps,
                      metric=metric, pgditer=pgditer, verbose=True)
    print(advrank)
    xr, r, sumorig, sumadv = advrank(images, labels, candi)
    print('DEBUG: sumorig', sumorig)
    print('DEBUG: sumadv', sumadv)
    if eps < 1e-3:
        assert(abs(sumorig['loss'] - sumadv['loss']) < 1e-3)  # sanity test
    assert(r.abs().max() < eps + 1e-4)  # sanity test
    if eps > 1e-3:
        assert(r.abs().max() >= 1. / 255.)  # sanity test


@pytest.mark.parametrize('W, pm, metric, pgditer, eps',
                         it.product((1, 2, 5), ('+', '-'), ('C', 'E', 'N'), (1, 4), (0., 0.1)))
def test_ca(W, pm, metric, pgditer, eps):
    # pylint: disable=unused-variable
    model = th.nn.Sequential(th.nn.Linear(8, 8))
    print(model)
    dataset = (th.rand(1000, 8), th.randint(0, 10, (1000,)))
    with th.no_grad():
        if metric in ('C', 'N'):
            candi = (F.normalize(model(dataset[0])), dataset[1])
        else:
            candi = (model(dataset[0]), dataset[1])
    print(candi)
    # select first 8 to perturb
    images, labels = dataset[0][:8], dataset[1][:8]

    advrank = AdvRank(model, attack_type='CA', eps=eps, W=W,
                      pm=pm, metric=metric, pgditer=pgditer, verbose=True)
    print(advrank)
    xr, r, sumorig, sumadv = advrank(images, labels, candi)
    print('DEBUG: sumorig', sumorig)
    print('DEBUG: sumadv', sumadv)
    if eps < 1e-3:
        assert(abs(sumorig['loss'] - sumadv['loss']) < 1e-3)  # sanity test
    assert(r.abs().max() < eps + 1e-4)  # sanity test
    if eps > 1e-3:
        assert(r.abs().mean() > 1e-3)  # sanity test


@pytest.mark.parametrize('attack_type, M, pm, metric, pgditer, eps',
                         it.product(('QA', 'SPQA'), (1, 2, 5), ('+', '-'), ('C', 'E', 'N'), (1, 4), (0., 0.1)))
def test_qa(attack_type, M, pm, metric, pgditer, eps):
    # pylint: disable=unused-variable
    model = th.nn.Sequential(th.nn.Linear(8, 8))
    print(model)
    dataset = (th.rand(1000, 8), th.randint(0, 10, (1000,)))
    with th.no_grad():
        if metric in ('C', 'N'):
            candi = (F.normalize(model(dataset[0])), dataset[1])
        else:
            candi = (model(dataset[0]), dataset[1])
    print(candi)
    # select first 8 to perturb
    images, labels = dataset[0][:8], dataset[1][:8]

    advrank = AdvRank(model, attack_type=attack_type, eps=eps,
                      M=M, pm=pm, metric=metric, pgditer=pgditer, verbose=True)
    print(advrank)
    xr, r, sumorig, sumadv = advrank(images, labels, candi)
    print('DEBUG: sumorig', sumorig)
    print('DEBUG: sumadv', sumadv)
    if eps < 1e-3:
        assert(abs(sumorig['loss'] - sumadv['loss']) < 1e-3)  # sanity test
    assert(r.abs().max() < eps + 1e-4)  # sanity test
    if eps > 1e-3:
        assert(r.abs().mean() > 1e-3)  # sanity test


@pytest.mark.parametrize('metric, pgditer, eps',
                         it.product(('N', 'E', 'C'), (1, 4), (0., 0.1)))
def test_gtm(metric: str, pgditer: int, eps: float):
    # pylint: disable=unused-variable
    model = th.nn.Sequential(th.nn.Linear(8, 8))
    print(model)
    dataset = (th.rand(1000, 8), th.randint(0, 10, (1000,)))
    with th.no_grad():
        if metric in ('C', 'N'):
            candi = (F.normalize(model(dataset[0])), dataset[1])
        else:
            candi = (model(dataset[0]), dataset[1])
    print(candi)
    # attack the model to reduce r@1
    images, labels = dataset[0][:8], dataset[1][:8]
    advrank = AdvRank(model, attack_type='GTM', eps=eps,
                      metric=metric, pgditer=pgditer, verbose=True)
    print(advrank)
    xr, r, sumorig, sumadv = advrank(images, labels, candi)
    print('DEBUG: sumorig', sumorig)
    print('DEBUG: sumadv', sumadv)
    if eps == 0.:
        assert(abs(sumorig['r@1'] - sumadv['r@1']) < 1e3)
    # else:
    #    assert(sumorig['r@1'] >= sumadv['r@1'])
