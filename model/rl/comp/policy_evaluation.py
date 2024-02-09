#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging

import numpy as np
from collections import defaultdict, OrderedDict

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


def defaultdict_float():
  return defaultdict(float)


# Iterative Policy Evaluation.
def ipe(tp, rf, info, T=int(1e4), eps=1e-5, vf=None):
  obss = info['obss']
  acts = info['acts']
  n_states = info['n_states']
  n_actions = info['n_actions']

  # probses = info['probses']
  tparray = info['tparray']
  # successor = info['successor']
  predecessor = info['predecessor']
  taken = info['taken']
  sas = info['sas']
  oaprob = info['oaprob']

  gamma = info['gamma']

  if vf is None or len(vf) == 0:
    vf = defaultdict(float)
    for obsk in obss:
      vf[obsk] = np.random.rand()

  for t in range(T):
    delta = 0.0
    for obsi in range(n_states):
      obsk = obss[obsi]
      oldv = vf[obsk]
      tpp = tparray[obsi]
      v = 0.0
      for acti in taken[obsi]:
        actk = acts[acti]
        s = rf[obsk][actk]
        tppa = tpp[acti]
        for nobsi in sas[obsi][acti]:
        # for nobsi in successor[obsi]:
          nobsk = obss[nobsi]
          s += tppa[nobsi] * (gamma * vf[nobsk])
        prob = oaprob[obsi, acti]
        v += prob * s
      # logging.error('%f, %f, %f' % (oldv, v, np.sum(oaprob[obsi])))
      vf[obsk] = v
      delta = max(delta, np.abs(oldv - v))
    if delta < eps:
      logging.error('%d, %f, %d, %d' % (t, delta, n_states, n_actions))
      break

  return vf


def get_qf_from_vf(tp, rf, vf, info):
  gamma = info['gamma']
  qf = defaultdict(defaultdict_float)
  for obsk in tp:
    tpp = tp[obsk]
    for actk in tpp:
      s = rf[obsk][actk]
      tppa = tpp[actk]
      for nobsk in tppa:
        s += tppa[nobsk] * gamma * vf[nobsk]
      qf[obsk][actk] = s
  return qf


def get_advantage_function(vf, qf):
  af = defaultdict(defaultdict_float)
  for obsk in qf:
    # if obsk != (0.0, 0.0,):
    #   continue
    qfo = qf[obsk]
    for actk in qfo:
      af[obsk][actk] = qfo[actk] - vf[obsk]
  return af


def normalize_advantage_function(af):
  values = []
  for obsk in af:
    for actk in af[obsk]:
      values.append(af[obsk][actk])
  alladv = np.array(values)
  std = alladv.std()
  mean = alladv.mean()
  out = defaultdict(defaultdict_float)
  for obsk in af:
    for actk in af[obsk]:
      out[obsk][actk] = (af[obsk][actk] - mean) / std
  return out


def validate_qf_vf(tp, policy, vf, qf):
  obss = list(tp.keys())
  acts = set()
  for obsk in tp:
    for actk in tp[obsk]:
      acts.add(actk)
  acts = list(acts)

  oaprob = defaultdict(defaultdict_float)
  for obsk in obss:
    s = 0.0
    _, ps = policy.act([obsk])
    for actk in acts:
      prob = policy.prob_type.py_likelihood(actk, ps[0])
      oaprob[obsk][actk] = prob
      s += prob
    for actk in acts:
      oaprob[obsk][actk] = oaprob[obsk][actk] / s

  loss = 0.0
  inf = defaultdict(float)
  for obsk in qf:
    for actk in qf[obsk]:
      inf[obsk] += oaprob[obsk][actk] * qf[obsk][actk]
    loss += np.abs(inf[obsk] - vf[obsk])

  return loss
