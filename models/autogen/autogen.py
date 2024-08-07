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
import itertools as it
import rich
c = rich.get_console()

HM_TEMPLATE_RC2F2 = '''
from .. import rc2f2
class Model(rc2f2.Model):
    is_advtrain_hm = True
'''

HM_TEMPLATE_RRES18 = '''
from . import rres18
class Model(rres18.Model):
    is_advtrain_hm = True
'''

HM_TEMPLATES = [
    ('rc2f2', HM_TEMPLATE_RC2F2),
    ('rres18', HM_TEMPLATE_RRES18)
]

HARDNESS = ('r', 'm', 's', 'd', 'h')
HARDNESS_MAP = {'r': 'spc2-random', 'm': 'spc2-semihard',
                's': 'spc2-softhard', 'd': 'spc2-distance', 'h': 'spc2-hard'}


def write_model_config(filename, template, grad, hm, srch, desth):
    pad = '    '
    end = '\n'
    with open(filename, 'wt') as f:
        f.write(template)
        f.write(pad + 'hm_spec = {' + end)
        f.write(pad * 2 + f"'hm': '{hm.upper()}'," + end)
        f.write(pad * 2 + f"'gradual': '{str(g)}'," + end)
        f.write(pad * 2 + f"'srch': '{HARDNESS_MAP[srch]}'," + end)
        f.write(pad * 2 + f"'desth': '{HARDNESS_MAP[desth]}'," + end)
        f.write(pad + '}' + end)


for (name, template) in HM_TEMPLATES:
    for g in (False, True):
        for hm in ('kl', 'l2'):
            for (srch, desth) in it.product(HARDNESS, HARDNESS):
                filename = name
                if g:
                    filename += 'g'
                filename += hm
                filename += srch
                filename += desth
                filename += '.py'
                with c.status('Creating ' + filename + ' ...'):
                    write_model_config(filename, template, g, hm, srch, desth)
