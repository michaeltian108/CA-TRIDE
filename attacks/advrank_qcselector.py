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

# pylint: disable=no-member
import os
import torch as th
import numpy as np
import pytest
import itertools as it


class QCSelector(object):
    '''
    Select C / Q for adversarial ranking attacks
    '''

    def __init__(self, attack_type: str, M: int = None,
                 W: int = None, SP: bool = False):
        if attack_type in ('ES', 'GTM', 'GTT'):
            assert((M is None) and (W is None))
        elif attack_type == 'FOA':
            assert((M is not None) and (M >= 2))
        elif attack_type in ('CA+', 'CA-'):
            assert((M is None) and (W is not None))
        elif attack_type in ('QA+', 'QA-'):
            assert((M is not None) and (W is None))
        else:
            pass
        self.attack_type = attack_type
        self.M = M
        self.W = W
        self.SP = SP
        # sensible choice due to SOP dataset property
        self.M_GT = 5  # [2002.11293 Locked. Don't modify!]
        self.map = {
            'ES': self._sel_es,
            'FOA': self._sel_foa,
            'CA+': self._sel_caplus,
            'CA-': self._sel_caminus,
            'QA+': self._sel_qaplus,
            'QA-': self._sel_qaminus,
            'GTM': self._sel_gtm,
            'GTT': self._sel_gtt,
            'TMA': self._sel_tma,
        }

    def __call__(self, *argv, **kwargs):
        with th.no_grad():
            ret = self.map[self.attack_type](*argv, **kwargs)
        return ret

    def _sel_es(self, dist, candi):
        # -- [orig] untargeted attack, UT
        # >> the KNN accuracy, or say the Recall@1
        return None

    def _sel_foa(self, dist, candi):
        # == configuration for FOA:M=2, M>2
        M_GT = self.M_GT
        M = self.M

        # == select the M=2 candidates. note, x1 is closer to q than x2
        if self.M == 2:

            if True:
                # local sampling (default)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost + 1, dim=1,
                                  largest=False)[1][:, 1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost)
                                 for j in range(topxm.shape[0])])
                msample = th.stack([topxm[i][np.sort(sel[i, :M])]
                                    for i in range(topxm.shape[0])])
                if self.SP:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i, M:])[:M_GT]]
                                        for i in range(topxm.shape[0])])
            else:
                # global sampling
                distsort = dist.sort(dim=1)[1]  # [output_0, candi_0]
                mpairs = th.randint(
                    candi[0].shape[0], (dist.shape[0], M)).sort(
                    dim=1)[0]  # [output_0, M]
                msample = th.stack([distsort[i, mpairs[i]]
                                    for i in range(dist.shape[0])])  # [output_0, M]
                if self.SP:
                    # [output_0, M_GT]
                    mgtruth = dist.topk(
                        M_GT + 1, dim=1, largest=False)[1][:, 1:]
            embpairs = candi[0][msample, :]  # [output_0, M, output_1]
            if self.SP:
                embgts = candi[0][mgtruth, :]  # [output_0, M_GT, output_1]

        # == select M>2 candidates, in any order
        elif self.M > 2:

            if True:
                # local sampling (from topmost samples)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost + 1, dim=1,
                                  largest=False)[1][:, 1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost)
                                 for j in range(topxm.shape[0])])
                msample = th.stack([topxm[i][sel[i, :M]]
                                    for i in range(topxm.shape[0])])
                if self.SP:
                    mgtruth = th.stack([topxm[i][np.sort(sel[i, M:])[:M_GT]]
                                        for i in range(topxm.shape[0])])
            else:
                # global sampling
                msample = th.randint(
                    candi[0].shape[0], (dist.shape[0], M))  # [output_0, M]
                if self.SP:
                    mgtruth = dist.topk(
                        M_GT + 1, dim=1, largest=False)[1][:, 1:]
            embpairs = candi[0][msample, :]  # [output_0, M, output_1]
            if self.SP:
                embgts = candi[0][mgtruth, :]  # [output_0, M_GT, output_1]

        # return selections
        if self.SP:
            return (embpairs, msample, embgts, mgtruth)
        else:
            return (embpairs, msample)

    def _sel_caplus(self, dist, candi):
        # -- [orig] candidate attack, W=?
        # >> select W=? attacking targets
        if 'global' == os.getenv('SAMPLE', 'global'):
            msample = th.randint(
                candi[0].shape[0], (dist.shape[0], self.W))  # [output_0, W]
        elif 'local' == os.getenv('SAMPLE', 'global'):
            local_lb = int(candi[0].shape[0] * 0.01)
            local_ub = int(candi[0].shape[0] * 0.05)
            topxm = dist.topk(local_ub + 1, dim=1, largest=False)[1][:, 1:]
            sel = np.random.randint(
                local_lb, local_ub, (dist.shape[0], self.W))
            msample = th.stack([topxm[i][sel[i]]
                                for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]
        return (embpairs, msample)

    def _sel_caminus(self, dist, candi):
        # these are not the extremely precise topW queries but an approximation
        # select W candidates from the topmost samples
        topmost = int(candi[0].size(0) * 0.01)
        if int(os.getenv('VIS', 0)) > 0:
            topmost = int(candi[0].size(0) * 0.0003)
        topxm = dist.topk(topmost +
                          1, dim=1, largest=False)[1][:, 1:]  # [output_0, W]
        sel = np.random.randint(0, topmost, [topxm.shape[0], self.W])
        msample = th.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]  # [output_0, W, output_1]
        return (embpairs, msample)

    def _sel_qaplus(self, dist, candi):
        M_GT = self.M_GT
        M = self.M
        # random sampling from populationfor QA+
        if 'global' == os.getenv('SAMPLE', 'global'):
            msample = th.randint(
                candi[0].shape[0], (dist.shape[0], M))  # [output_0,M]
        elif 'local' == os.getenv('SAMPLE', 'global'):
            local_lb = int(candi[0].shape[0] * 0.01)
            local_ub = int(candi[0].shape[0] * 0.05)
            topxm = dist.topk(local_ub + 1, dim=1, largest=False)[1][:, 1:]
            sel = np.random.randint(local_lb, local_ub, (dist.shape[0], M))
            msample = th.stack([topxm[i][sel[i]]
                                for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]
        if self.SP:
            mgtruth = dist.topk(
                M_GT + 1, dim=1, largest=False)[1][:, 1:]  # [output_0, M]
            embgts = candi[0][mgtruth, :]
        # return the selections
        if self.SP:
            return (embpairs, msample, embgts, mgtruth)
        else:
            return (embpairs, msample)

    def _sel_qaminus(self, dist, candi) -> tuple:
        M_GT = self.M_GT
        M = self.M
        # random sampling from top-3M for QA-
        topmost = int(candi[0].size(0) * 0.01)
        if int(os.getenv('VIS', 0)) > 0:
            topmost = int(candi[0].size(0) * 0.0003)
        topxm = dist.topk(topmost + 1, dim=1, largest=False)[1][:, 1:]
        sel = np.vstack([np.random.permutation(topmost)
                         for i in range(dist.shape[0])])
        msample = th.stack([topxm[i][sel[i, :M]]
                            for i in range(topxm.shape[0])])
        if self.SP:
            mgtruth = th.stack([topxm[i][np.sort(sel[i, M:])[:M_GT]]
                                for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]
        if self.SP:
            embgts = candi[0][mgtruth, :]
        # return selections
        if self.SP:
            return (embpairs, msample, embgts, mgtruth)
        else:
            return (embpairs, msample)

    def _sel_tma(self, dist, candi) -> tuple:
        # random sampling is enough. -- like QA+
        idxrand = th.randint(candi[0].size(0), (dist.size(0),))
        embrand = candi[0][idxrand, :]
        return (embrand, idxrand)

    def _sel_gtm(self, dist, candi, loc_self) -> tuple:
        # top-1 matching and top-1 unmatching
        d = dist.clone()
        # filter out the query itself
        d[range(len(loc_self)), loc_self] = 1e38
        argsort = d.argsort(dim=1, descending=False)
        argrank = argsort.argsort(dim=1, descending=False)
        mylabel = candi[1][argsort[:, 0]].view(-1, 1)
        mask_match = (candi[1].view(1, -1) == mylabel)
        mask_unmatch = (candi[1].view(1, -1) != mylabel)
        fstmatch = th.stack([argrank[i, mask_match[i]].argmin()
                             for i in range(mask_match.size(0))])
        fstunmatch = th.stack([argrank[i, mask_unmatch[i]].argmin()
                               for i in range(mask_unmatch.size(0))])
        ret = ((candi[0][fstmatch, None, :], fstmatch),
               (candi[0][fstunmatch, None, :], fstunmatch),
               (candi[0][loc_self, None, :], loc_self))
        return ret

    def _sel_gtt(self, dist, candi, loc_self) -> tuple:
        # non-self top-2
        d = dist.clone()
        d[range(len(loc_self)), loc_self] = 1e38
        idtop2 = d.topk(2, dim=-1, largest=False)[1]
        fstidm = idtop2[:, 0]
        fstidum = idtop2[:, 1]
        ret = ((candi[0][fstidm, None, :], fstidm),
               (candi[0][fstidum, None, :], fstidum),
               (candi[0][loc_self, None, :], loc_self))
        return ret


def test_qcs_es():
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector('ES', None, None, False)(dist, candi)


@pytest.mark.parametrize('W, pm', it.product((2, 5, 10), ('+', '-')))
def test_qcs_ca(pm: str, W: int):
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector(f'CA{pm}', None, W, False)(dist, candi)


@pytest.mark.parametrize('pm, M', it.product(('+', '-'), (2, 5, 10)))
def test_qcs_qa(pm: str, M: int):
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector(f'QA{pm}', M, None, False)(dist, candi)


@pytest.mark.parametrize('pm, M', it.product(('+', '-'), (2, 5, 10)))
def test_qcs_spqa(pm: str, M: int):
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector(f'QA{pm}', M, None, True)(dist, candi)


@pytest.mark.parametrize('M', (2, 5, 10))
def test_qcs_foa(M: int):
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector('FOA', M, None, False)(dist, candi)


@pytest.mark.parametrize('M', (2, 5, 10))
def test_qcs_spfoa(M: int):
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector('FOA', M, None, True)(dist, candi)


def test_gtm():
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    loc_self = th.randint(128, (10,))
    _ = QCSelector('GTM', None, None)(dist, candi, loc_self)


def test_gtt():
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    loc_self = th.randint(128, (10,))
    _ = QCSelector('GTT', None, None)(dist, candi, loc_self)


def test_tma():
    dist = th.rand(10, 128)
    candi = (th.rand(128, 128), th.randint(5, (128,)))
    _ = QCSelector('TMA', None, None)(dist, candi)
