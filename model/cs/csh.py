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


class CSHMin(cs_lib.ClientSelectionBase):

  def __init__(self, clients, seed, **kwargs):
    super(CSHMin, self).__init__(clients, seed)
    # self.positives = [i for i in range(len(clients))
    #                   if self.bs[i] >= 0]
    # self.negatives = [i for i in range(len(clients))
    #                   if self.bs[i] < 0]
    # self.stochastic = kwargs['stochastic']

  def select_clients(self, num_clients, indices_cands):
    num_clients = min(num_clients, len(indices_cands))
    # indices = np.argsort([self.bs[i] for i in indices_cands])
    indices = np.argsort(self.bs)
    indices = indices[:num_clients]
    return ([indices_cands[i] for i in indices],
            [self.clients[indices_cands[i]] for i in indices])

    num_clients = min(num_clients, len(self.clients))
    indices = np.argsort(self.bs)
    indices = indices[:num_clients]
    return indices, [self.clients[i] for i in indices]

    np.random.seed(round_id)
    if len(self.positives) < num_clients:
      num_clients = num_clients - len(self.positives)
      indices = np.random.choice(
          range(len(self.negatives)), num_clients, replace=False)
      return [self.negatives[i] for i in indices] + self.positives, [self.clients[self.negatives[i]] for i in indices] + [self.clients[i] for i in self.positives]
    indices = np.random.choice(
        range(len(self.positives)), num_clients, replace=False)
    return [self.positives[i] for i in indices], [self.clients[self.positives[i]] for i in indices]


class CSHMax(cs_lib.ClientSelectionBase):

  def __init__(self, clients, seed, **kwargs):
    super(CSHMax, self).__init__(clients, seed)
    # self.positives = [i for i in range(len(clients))
    #                   if self.bs[i] >= 0]
    # self.negatives = [i for i in range(len(clients))
    #                   if self.bs[i] < 0]
    # self.stochastic = kwargs['stochastic']

  def select_clients(self, num_clients, indices_cands):
    num_clients = min(num_clients, len(indices_cands))
    # indices = np.argsort([self.bs[i] for i in indices_cands])
    indices = np.argsort(self.bs)
    indices = indices[len(indices) - num_clients:]
    return ([indices_cands[i] for i in indices],
            [self.clients[indices_cands[i]] for i in indices])

    num_clients = min(num_clients, len(self.clients))
    indices = np.argsort(self.bs)
    indices = indices[len(indices) - num_clients:]
    return indices, [self.clients[i] for i in indices]

    np.random.seed(round_id)
    if len(self.negatives) < num_clients:
      num_clients = num_clients - len(self.negatives)
      indices = np.random.choice(
          range(len(self.positives)), num_clients, replace=False)
      return [self.positives[i] for i in indices] + self.negatives, [self.clients[self.positives[i]] for i in indices] + [self.clients[i] for i in self.negatives]
    indices = np.random.choice(
        range(len(self.negatives)), num_clients, replace=False)
    return [self.negatives[i] for i in indices], [self.clients[self.negatives[i]] for i in indices]
