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
import sys
import consle
sys.path.append('.')
consle.AdvRank(sys.argv[1:])
