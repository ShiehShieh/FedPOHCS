from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import random
import numpy as np

from collections import defaultdict

from mujoco_py.builder import MujocoException

import model.cs.cs as cs_lib


class GradNormSelection(cs_lib.ClientSelectionBase):

  def __init__(self, clients, seed, **kwargs):
    super(GradNormSelection, self).__init__(clients, seed)

  def select_clients(self, num_clients, indices_cands):
    num_clients = min(num_clients, len(indices_cands))
    indices = np.argsort(self.gradnorms)
    # Selecting clients with highest gradient norm.
    indices = indices[len(indices) - num_clients:]
    return ([indices_cands[i] for i in indices],
            [self.clients[indices_cands[i]] for i in indices])
