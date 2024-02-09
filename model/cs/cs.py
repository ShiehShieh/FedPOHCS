from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import random
import numpy as np

from collections import defaultdict


class ClientSelectionBase(object):

  def __init__(self, clients, seed, **kwargs):
    self.clients = clients
    np.random.seed(seed)
    # make sure for each comparison, we are selecting the same clients each round
    # np.random.seed(round_id + self.seed)
    # np.random.seed(round_id)
    # np.random.seed(round_id + 1000)  #
    # np.random.seed(round_id + 10000)  #

  def select_candidates(self, num_cands):
    num_cands = min(num_cands, len(self.clients))
    indices = np.random.choice(
        range(len(self.clients)), num_cands, replace=False)
    return indices, [self.clients[i] for i in indices]

  def register_heterogeneity_level(self, bs):
    self.bs = np.array(bs)

  def register_obj(self, objs):
    self.objs = np.array(objs)

  def register_gradnorm(self, gradnorms):
    self.gradnorms = np.array(gradnorms)

  def select_clients(self, num_clients, indices_cands):
    raise NotImplementedError
