from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import random
import numpy as np

from collections import defaultdict

import model.cs.cs as cs_lib


class Random(cs_lib.ClientSelectionBase):

  def __init__(self, clients, seed, **kwargs):
    super(Random, self).__init__(clients, seed)

  # def select_candidates(self, num_cands):
  #   return list(range(len(self.clients))), self.clients

  def select_clients(self, num_clients, indices_cands):
    num_clients = min(num_clients, len(indices_cands))
    # num_clients = min(num_clients, len(self.clients))
    # make sure for each comparison, we are selecting the same clients each round
    # np.random.seed(round_id + self.seed)
    # np.random.seed(round_id)
    # np.random.seed(round_id + 1000)  #
    # np.random.seed(round_id + 10000)  #

    # NOTE(XIE,Zhijie): This is a more realistic implementation.
    # indices = np.random.choice(
    #     range(len(self.clients)), num_clients, replace=False)
    # return indices, [self.clients[i] for i in indices]

    # NOTE(XIE,Zhijie): This is to keep it consistent with other client
    # selection schemes.
    # np.random.rand(1)
    # indices = list(range(num_clients))
    indices = np.random.choice(
        range(len(indices_cands)), num_clients, replace=False)
    return ([indices_cands[i] for i in indices],
            [self.clients[indices_cands[i]] for i in indices])
