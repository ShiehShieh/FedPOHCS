#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from absl import logging

import numpy as np
from itertools import product
from collections import defaultdict, OrderedDict

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import model.rl.comp.policy_evaluation as pe_lib


def round_resolution(x, resolution):
  return np.round(np.round(x / resolution) * resolution, 2)
  return np.round(x / resolution) * resolution


def round_obs(obs):
  o = np.zeros(shape=(2,))
  # o = np.copy(obs)
  # o[0] = round_resolution(obs[0], 0.1)
  # o[1] = round_resolution(obs[1], 0.001)
  o[0] = round_resolution(obs[0], 0.3)
  o[1] = round_resolution(obs[1], 0.3)
  return o


def round_act(act):
  # o = np.copy(act)
  o = np.zeros(shape=(1,))
  o[0] = round_resolution(act[0], 0.1)
  return o


def update_initial_state_distribution(ism, trajectories):
  '''
  ism: [s] -> float
  '''
  for i, trajectory in enumerate(trajectories):
    for j, obs in enumerate(trajectory['observations'][:1]):
      # obs = round_resolution(obs, 0.2)
      obs = round_obs(obs)
      obsk = tuple(obs.tolist())
      ism[obsk] += 1.0


def update_transition_probability(tpm, trajectories):
  '''
  tpm: [s, a, s'] -> float
  '''
  for i, trajectory in enumerate(trajectories):
    tactions = trajectory['actions']
    tnobs = trajectory['next_observations']
    for j, obs in enumerate(trajectory['observations']):
      act = tactions[j]
      nobs = tnobs[j]
      # obs = round_resolution(obs, 0.2)
      # act = round_resolution(act, 0.1)
      # nobs = round_resolution(nobs, 0.2)
      obs = round_obs(obs)
      act = round_act(act)
      nobs = round_obs(nobs)
      obsk = tuple(obs.tolist())
      actk = tuple(act.tolist())
      nobsk = tuple(nobs.tolist())
      tpm[obsk][actk][nobsk] += 1.0


def update_reward_function(rm, trajectories):
  '''
  rm: [s, a, r] -> float
  '''
  for i, trajectory in enumerate(trajectories):
    tactions = trajectory['actions']
    trewards = trajectory['reward']
    for j, obs in enumerate(trajectory['observations']):
      act = tactions[j]
      # r = np.round(trajectory['reward'][j], 2)
      r = trewards[j]
      # obs = round_resolution(obs, 0.2)
      # act = round_resolution(act, 0.1)
      obs = round_obs(obs)
      act = round_act(act)
      obsk = tuple(obs.tolist())
      actk = tuple(act.tolist())
      rmoa = rm[obsk][actk]
      rmoa[0] += r
      rmoa[1] += 1.0


def defaultdict_float():
  return defaultdict(float)


def defaultdict_defaultdict_float():
  return defaultdict(defaultdict_float)


def defaultdict_set():
  return defaultdict(set)


def get_initial_state_distribution(ism):
  isd = defaultdict(float)
  s = 0.0
  for obsk in ism:
    s += ism[obsk]
  for obsk in ism:
    isd[obsk] = ism[obsk] / s
  return isd


def get_transition_probability(tpm):
  o = defaultdict(defaultdict_defaultdict_float)
  for obsk in tpm:
    for actk in tpm[obsk]:
      s = 0.0
      for nobsk in tpm[obsk][actk]:
        s += tpm[obsk][actk][nobsk]
      for nobsk in tpm[obsk][actk]:
        o[obsk][actk][nobsk] = tpm[obsk][actk][nobsk] / s
  return o


def get_reward_function(rm):
  o = defaultdict(defaultdict_float)
  for obsk in rm:
    for actk in rm[obsk]:
      (s, cnt) = rm[obsk][actk]
      o[obsk][actk] = s / cnt
  return o


def get_svf_by_maxent(tp, isd, info, T=500):
  obss = info['obss']
  acts = info['acts']
  n_states = info['n_states']
  n_actions = info['n_actions']

  # probses = info['probses']
  tparray = info['tparray']
  # successor = info['successor']
  predecessor = info['predecessor']
  taken = info['taken']
  ssa = info['ssa']
  oaprob = info['oaprob']

  d = np.zeros((n_states, T))
  for obsi in range(n_states):
    d[obsi, 0] = isd[obss[obsi]]

  for t in range(1, T):
    for obsi in range(n_states):
      for pobsi in predecessor[obsi]:
        pd = d[pobsi, t - 1]
        if pd == 0:
          continue
        tpp = tparray[pobsi]
        for acti in ssa[pobsi][obsi]:
        # for acti in taken[pobsi]:
          tppao = tpp[acti, obsi]
          # if tppao == 0:
          #   continue
          prob = oaprob[pobsi, acti]
          d[obsi, t] += pd * prob * tppao

  total = d.sum()
  o = defaultdict(float)
  for obsi in range(n_states):
    obsk = obss[obsi]
    # if obsk != (0.0, 0.0,):
    #   continue
    o[obsk] = d[obsi].sum() / total
  return o


def prepare(tp, policy):
  obss = list(tp.keys())
  acts = set()
  for obsk in tp:
    for actk in tp[obsk]:
      acts.add(actk)
  acts = list(acts)
  n_states = len(obss)
  n_actions = len(acts)

  probses = []
  for idx in range(max(len(obss) // 128, 1)):
    start = idx * 128
    end = start + 128
    if idx == (len(obss) // 128) - 1:
      end = len(obss)
    _, ps = policy.act(obss[start:end])
    for i, obsk in enumerate(obss[start:end]):
      probses.append(ps[i])

  tparray = np.zeros((n_states, n_actions, n_states))
  successor = defaultdict(set)
  predecessor = defaultdict(set)
  taken = defaultdict(set)
  ssa = defaultdict(defaultdict_set)
  sas = defaultdict(defaultdict_set)
  for obsi in range(n_states):
    obsk = obss[obsi]
    tpo = tp[obsk]
    for acti in range(n_actions):
      actk = acts[acti]
      if actk not in tpo:
        continue
      tpoa = tpo[actk]
      taken[obsi].add(acti)
      for nobsi in range(n_states):
        nobsk = obss[nobsi]
        if nobsk not in tpoa:
          continue
        tparray[obsi, acti, nobsi] = tpoa[nobsk]
        # successor[obsi].add(nobsi)
        predecessor[nobsi].add(obsi)
        ssa[obsi][nobsi].add(acti)
        sas[obsi][acti].add(nobsi)

  oaprob = np.zeros((n_states, n_actions))
  for obsi in range(n_states):
    for acti in range(n_actions):
      actk = acts[acti]
      prob = policy.prob_type.py_likelihood(actk, probses[obsi])
      oaprob[obsi, acti] = prob
    oaprob[obsi] = oaprob[obsi] / oaprob[obsi].sum()

  return {
      'obss': obss,
      'acts': acts,
      'n_states': n_states,
      'n_actions': n_actions,
      # 'probses': probses,
      'tparray': tparray,
      # 'successor': successor,
      'predecessor': predecessor,
      'taken': taken,
      'ssa': ssa,
      'sas': sas,
      'oaprob': oaprob,
      'gamma': policy.future_discount,
  }


def find_svf_v2(isd, tp, rf, vf, info):
  d = get_svf_by_maxent(tp, isd, info)
  vf = pe_lib.ipe(tp, rf, info, vf=vf, eps=1e-5)
  qf = pe_lib.get_qf_from_vf(tp, rf, vf, info)
  af = pe_lib.get_advantage_function(vf, qf)
  af = pe_lib.normalize_advantage_function(af)
  # start = time.time()
  # err = pe_lib.validate_qf_vf(tp, policy, vf, qf)
  # logging.error('validate_qf_vf: %s' % (time.time() - start,))
  # logging.error(err)
  return d, af, vf


def find_discounted_svf(n_states, trajectories, svf_m=None, gamma=1.0):
  # Continuous state space.
  # OrderedDict(sorted(d.items()))
  if n_states == -1:
    seq_len = [t['observations'].shape[0] for t in trajectories]
    max_seq_len = np.max(seq_len)
    mask = np.array([[float(j < sl) for j in range(max_seq_len)]
                     for i, sl in enumerate(seq_len)])
    # pr = mask / mask.sum(axis=0)
    pr = mask / mask.sum()

    d = defaultdict(float)
    a = defaultdict(lambda: defaultdict(float))
    acnt = defaultdict(lambda: defaultdict(float))
    summation = 0.0
    for i, trajectory in enumerate(trajectories):
      for j, obs in enumerate(trajectory['observations']):
        act = trajectory['actions'][j]
        act = np.round(act, decimals=1)
        obs = np.round(obs, decimals=1)
        # # Reacher.
        # act = np.round(act, decimals=1)
        # obs = np.round(obs, decimals=0)
        # # MountainCarContinuous.
        # act = np.round(act, decimals=1)
        # obs = np.round(obs, decimals=2)
        # act = round_resolution(act, resolution=0.1)
        # obs = round_resolution(obs, resolution=0.01)
        # obsk = ','.join(map(str, obs.tolist()))
        # actk = ','.join(map(str, act.tolist()))
        obsk = tuple(obs.tolist())
        actk = tuple(act.tolist())
        d[obsk] += pow(gamma, j) * pr[i, j]
        # a[obsk][actk] += trajectory['prob_advs'][j]
        a[obsk][actk] += trajectory['advantages'][j]
        acnt[obsk][actk] += 1
        summation += d[obsk]
    for k, v in d.items():
      # d[k] = (1 - gamma) * v
      d[k] = (1 / summation) * v
    for k, kv in a.items():
      for kk, vv in kv.items():
        a[k][kk] /= acnt[k][kk]

    return d, a


def find_svf(n_states, trajectories, svf_m=None):
    """
    Find the state vistiation frequency from trajectories.
    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """
    # Continuous state space.
    if n_states == -1:
      m = defaultdict(float)
      if svf_m is not None:
        m = svf_m

      for trajectory in trajectories:
        for obs in trajectory['observations']:
          obs = np.round(obs, decimals=0)
          m[tuple(obs.tolist())] += 1.0

      return np.array([v for k, v in m.items()]) / float(len(trajectories)), m

    # Finite state space.
    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for obs in trajectory['observations']:
            svf[obs] += 1

    svf /= trajectories.shape[0]

    return svf


def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)
