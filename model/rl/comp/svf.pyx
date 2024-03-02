# cython: language_level=3
# distutils: language = c++
from __future__ import absolute_import, division, print_function

from absl import logging

import time
import numpy as np
from collections import defaultdict, OrderedDict

from cython.parallel import prange
from cython import boundscheck, wraparound
from cython.operator cimport dereference as deref, preincrement as inc

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from declarations cimport *


RESOLUTION = 0.1
CLIP = True


def set_resolution(resolution):
  global RESOLUTION
  RESOLUTION = resolution


def set_clip(clip):
  global CLIP
  CLIP = clip


def defaultdict_float():
  return defaultdict(float)


def defaultdict_defaultdict_float():
  return defaultdict(defaultdict_float)


def defaultdict_set():
  return defaultdict(set)


def round_resolution(x, resolution):
  return np.round(np.round(x / resolution) * resolution, 2)
  return np.round(x / resolution) * resolution


def round_obs(obs, resolution):
  o = np.zeros(shape=obs.shape)
  for i, ob in enumerate(obs):
    # o[i] = round_resolution(ob, 0.5) # mcc fuzzy.
    # o[i] = round_resolution(ob, 0.1) # mcc.

    # o[i] = round_resolution(ob, 0.4) # hopper.

    o[i] = round_resolution(ob, resolution)
  return o


def round_act(act, resolution, clip):
  o = np.zeros(shape=act.shape)
  for i, a in enumerate(act):
    # o[i] = round_resolution(a, 0.1) # mcc.
    # o[i] = round_resolution(a, 0.05) # mcc one-phase.

    # a = max(min(a, 1.0), -1.0) # hopper.
    # o[i] = round_resolution(a, 0.4) # hopper.

    if clip:
      a = max(min(a, 1.0), -1.0)
    o[i] = round_resolution(a, resolution)
  return o


def update_initial_state_distribution(ism, trajectories):
  '''
  ism: [s] -> float
  '''
  global RESOLUTION
  for i, trajectory in enumerate(trajectories):
    for j, obs in enumerate(trajectory['observations'][:1]):
      obs = round_obs(obs, RESOLUTION)
      obsk = tuple(obs.tolist())
      ism[obsk] += 1.0


def update_transition_probability(tpm, trajectories):
  '''
  tpm: [s, a, s'] -> float
  '''
  global RESOLUTION
  global CLIP
  for i, trajectory in enumerate(trajectories):
    tactions = trajectory['actions']
    tnobs = trajectory['next_observations']
    for j, obs in enumerate(trajectory['observations']):
      act = tactions[j]
      nobs = tnobs[j]
      obs = round_obs(obs, RESOLUTION)
      act = round_act(act, RESOLUTION, CLIP)
      nobs = round_obs(nobs, RESOLUTION)
      obsk = tuple(obs.tolist())
      actk = tuple(act.tolist())
      nobsk = tuple(nobs.tolist())
      tpm[obsk][actk][nobsk] += 1.0


def update_reward_function(rm, trajectories):
  '''
  rm: [s, a, r] -> float
  '''
  global RESOLUTION
  global CLIP
  for i, trajectory in enumerate(trajectories):
    tactions = trajectory['actions']
    trewards = trajectory['reward']
    for j, obs in enumerate(trajectory['observations']):
      act = tactions[j]
      r = trewards[j]
      obs = round_obs(obs, RESOLUTION)
      act = round_act(act, RESOLUTION, CLIP)
      obsk = tuple(obs.tolist())
      actk = tuple(act.tolist())
      rmoa = rm[obsk][actk]
      rmoa[0] += r
      rmoa[1] += 1.0


def get_initial_state_distribution(ism):
  isd = defaultdict(float)
  s = 0.0
  for obsk in ism:
    s += ism[obsk]
  for obsk in ism:
    isd[obsk] = ism[obsk] / s
  return dict(isd)


def get_transition_probability(tpm):
  o = defaultdict(defaultdict_defaultdict_float)
  for obsk in tpm:
    for actk in tpm[obsk]:
      s = 0.0
      for nobsk in tpm[obsk][actk]:
        s += tpm[obsk][actk][nobsk]
      for nobsk in tpm[obsk][actk]:
        o[obsk][actk][nobsk] = tpm[obsk][actk][nobsk] / s
      o[obsk][actk] = dict(o[obsk][actk])
    o[obsk] = dict(o[obsk])
  return dict(o)


def get_reward_function(rm):
  o = defaultdict(defaultdict_float)
  total_cnt = 0.0
  pair_nums = 0.0
  for obsk in rm:
    for actk in rm[obsk]:
      (s, cnt) = rm[obsk][actk]
      o[obsk][actk] = s / cnt
      total_cnt += cnt
      pair_nums += 1.0
    o[obsk] = dict(o[obsk])
  logging.error('total count: %f, average count per pair: %f' % (total_cnt, total_cnt / pair_nums,))
  return dict(o)


# Vectorized equivalent to multivariate_normal.pdf.
def mvn_pdf(x, p, pt):
  d = pt.d
  # mean = np.array([p[:d]] * x.shape([0]))
  # std = np.array([p[d:]] * x.shape([0]))
  # mean = np.asarray(mean, dtype=float)
  # cov = np.asarray(cov, dtype=float)

  # mean = [p[:d]] * x.shape[0]
  # # std = [p[d:]] * x.shape([0])
  # # cov = np.einsum('ji,ki->jik', std, np.eye(d, dtype=float))
  # cov = [np.diag(p[d:])] * x.shape[0]

  mean = p[:d]
  # std = [p[d:]] * x.shape([0])
  # cov = np.einsum('ji,ki->jik', std, np.eye(d, dtype=float))
  cov = np.diag(p[d:])

  # assert x.ndim == 2 and x.shape[1] == d
  # assert mean.ndim == 2 and mean.shape[1] == d
  # assert cov.ndim == 3 and cov.shape[1:] == (d, d)

  s, u = np.linalg.eigh(cov)
  eps = 2.22e-10 * np.max(np.abs(s), axis=0)

  # # Each covariance matrix must be symmetric positive definite
  # assert np.all(abs(s) > eps[:, None])

  # # The rest of the code is unsafe if this is not true
  # # It enable further optimizations
  # assert np.allclose(u, np.identity(d))

  log_pdet = np.sum(np.log(s), axis=0)

  maha = np.sum((x - mean)**2.0 / s, axis=1)
  log_2pi = np.log(2.0 * np.pi)
  logpdf = -0.5 * (d * log_2pi + log_pdet + maha)

  return np.exp(logpdf)


def prepare(tp, policy, logger=None):
  # TODO(XIE,Zhijie): Trim down states and actions that are not visited
  # for a while.
  start = time.time()

  # obss = set(tp.keys())
  # nobss = set()
  # acts = set()
  # for obsk in tp:
  #   acts = acts.union(tp[obsk].keys())
  #   for actk in tp[obsk]:
  #     nobss = nobss.union(tp[obsk][actk].keys())
  # obss = obss.union(nobss)
  # del nobss

  obss = set(tp.keys())
  nobss = list()
  acts = list()
  for obsk in tp:
    acts.extend(tp[obsk].keys())
    for actk in tp[obsk]:
      nobss.extend(tp[obsk][actk].keys())
  obss = obss.union(set(nobss))
  acts = set(acts)
  del nobss

  obss = list(obss)
  acts = list(acts)
  n_states = len(obss)
  n_actions = len(acts)
  robss = {obss[obsi]: obsi for obsi in range(n_states)}
  racts = {acts[acti]: acti for acti in range(n_actions)}
  if logger:
    logging.error('obss & acts time: %f s. n_states: %d, n_actions: %d.' % (time.time() - start, n_states, n_actions))

  start1 = time.time()
  probses = []
  for idx in range(max(len(obss) // 512, 1)):
    start = idx * 512
    end = start + 512
    if idx == max(len(obss) // 512, 1) - 1:
      end = len(obss)
    _, ps = policy.act(obss[start:end])
    # for i, obsk in enumerate(obss[start:end]):
    #   probses.append(ps[i])
    probses.extend(ps)
  if logger:
    logging.error('probses time: %f s.' % (time.time() - start1,))

  start = time.time()
  # tparray = np.zeros((n_states, n_actions, n_states))
  # tparray = [defaultdict(lambda: np.zeros(n_states))
  #     for i in range(n_states)]
  tparray = [defaultdict(lambda: dict()) for i in range(n_states)]
  successor = defaultdict(set)
  predecessor = defaultdict(set)
  taken = defaultdict(set)
  ssa = defaultdict(defaultdict_set)
  sas = defaultdict(defaultdict_set)
  for obsi in range(n_states):
    obsk = obss[obsi]
    tpo = tp.get(obsk, [])
    for actk in tpo:
      acti = racts[actk]
    # for acti in range(n_actions):
    #   actk = acts[acti]
    #   if actk not in tpo:
    #     continue
      tpoa = tpo[actk]
      taken[obsi].add(acti)
      for nobsk in tpoa:
        nobsi = robss[nobsk]
      # for nobsi in range(n_states):
      #   nobsk = obss[nobsi]
      #   if nobsk not in tpoa:
      #     continue
        tparray[obsi][acti][nobsi] = tpoa[nobsk]
        # successor[obsi].add(nobsi)
        predecessor[nobsi].add(obsi)
        ssa[obsi][nobsi].add(acti)
        sas[obsi][acti].add(nobsi)
  if logger:
    logging.error('tparray time: %f s.' % (time.time() - start,))

  for obsi in ssa:
    ssa[obsi] = dict(ssa[obsi])
  ssa = dict(ssa)
  for obsi in sas:
    sas[obsi] = dict(sas[obsi])
  sas = dict(sas)
  successor = dict(successor)
  predecessor = dict(predecessor)
  taken = dict(taken)

  start = time.time()
  oaprob = np.zeros((n_states, n_actions))
  npacts = np.array(acts)
  for obsi in range(n_states):
    p = probses[obsi]
    probs = policy.prob_type.py_likelihood(npacts, p)
    # probs = mvn_pdf(npacts, p, policy.prob_type)
    oaprob[obsi] = probs / probs.sum()
  #   # sm = 0.0
  #   # tpo = tp[obsk]
  #   # for acti in range(n_actions):
  #   #   actk = acts[acti]
  #   #   if actk not in tpo:
  #   #     # NOTE(XIE,Zhijie): No need to calculate those unvisited actions.
  #   #     continue
  #   #   prob = policy.prob_type.py_likelihood(actk, p)
  #   #   oaprob[obsi, acti] = prob
  #   #   sm += prob
  #   # oaprob[obsi] = oaprob[obsi] / sm
  if logger:
    logging.error('oaprob time: %f s.' % (time.time() - start,))

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


@boundscheck(False)
@wraparound(False)
cdef cmap[vector[double], double] get_svf_by_maxent(
    vector[vector[double]]& obss, vector[vector[double]]& acts,
    int& n_states, int& n_actions,
    # vector[vector[unordered_map[int, double]]]& tparray,
    vector[unordered_map[int, unordered_map[int, double]]]& tparray,
    unordered_map[int, vector[int]]& predecessor,
    unordered_map[int, unordered_map[int, vector[int]]]& ssa,
    vector[vector[double]]& oaprob,
    cmap[vector[double], double]& isd, int T=1000) nogil:

  cdef:
    vector[vector[double]] d = vector[vector[double]](n_states)
    int t
    double pd
    double tppao
    double prob
    # vector[vector[double]]* tpp
    # vector[double]* tppa
    # vector[unordered_map[int, double]]* tpp
    unordered_map[int, unordered_map[int, double]]* tpp
    unordered_map[int, unordered_map[int, double]].iterator itpp
    unordered_map[int, double]* tppa
    unordered_map[int, double].iterator itppa

  # d = np.zeros((n_states, T))
  for i in range(n_states):
    d[i].resize(T)
  for obsi in range(n_states):
    d[obsi][0] = 0
    iisd = isd.find(obss[obsi])
    if iisd != isd.end():
      d[obsi][0] = deref(iisd).second

  for t in range(1, T):
    for obsi in range(n_states):
      ipredecessor = predecessor.find(obsi)
      if ipredecessor == predecessor.end():
        continue
      for pobsi in deref(ipredecessor).second:
        pd = d[pobsi][t - 1]
        if pd == 0:
          continue
        tpp = &tparray[pobsi]
        issa = ssa.find(pobsi)
        if issa == ssa.end():
          continue
        jssa = deref(issa).second.find(obsi)
        if jssa == deref(issa).second.end():
          continue
        for acti in deref(jssa).second:
          # tppa = &deref(tpp)[acti]
          itpp = tpp.find(acti)
          if itpp == deref(tpp).end():
            continue
          tppa = &deref(itpp).second
          tppao = 0.0
          itppa = tppa.find(obsi)
          if itppa == tppa.end():
            continue
          tppao = deref(itppa).second
          prob = oaprob[pobsi][acti]
          d[obsi][t] += pd * prob * tppao

  cdef:
    double total = 0.0
    cmap[vector[double], double] o
    vector[double] rowssum

  rowssum.resize(n_states)
  for i in range(n_states):
    for j in range(T):
      total += d[i][j]
      rowssum[i] += d[i][j]
  for obsi in range(n_states):
    obsk = obss[obsi]
    o[obsk] = rowssum[obsi] / total

  return o


# Iterative Policy Evaluation.
@boundscheck(False)
@wraparound(False)
cdef cmap[vector[double], double] ipe(
    cmap[vector[double], cmap[vector[double], double]]& rf,
    cmap[vector[double], double]& vf,
    vector[vector[double]]& obss, vector[vector[double]]& acts,
    int& n_states, int& n_actions,
    # vector[vector[unordered_map[int, double]]]& tparray,
    vector[unordered_map[int, unordered_map[int, double]]]& tparray,
    unordered_map[int, vector[int]]& taken,
    unordered_map[int, unordered_map[int, vector[int]]]& sas,
    vector[vector[double]]& oaprob,
    double& gamma, int T=int(1e4), double eps=1e-5) nogil:

  # if vf is None or len(vf) == 0:
  #   vf = defaultdict(double)
  #   for obsk in obss:
  #     vf[obsk] = np.random.rand()

  cdef:
    double delta
    double v
    double nv
    double s
    double tppao
    double prob
    double oldv
    int obsi, acti, nobsi, t
    vector[double] obsk, actk
    # vector[vector[double]]* tpp
    # vector[double]* tppa
    # vector[unordered_map[int, double]]* tpp
    # unordered_map[int, double]* tppa
    # unordered_map[int, double].iterator itppa
    unordered_map[int, unordered_map[int, double]]* tpp
    unordered_map[int, unordered_map[int, double]].iterator itpp
    unordered_map[int, double]* tppa
    unordered_map[int, double].iterator itppa

  for t in range(T):
    delta = 0.0
    for obsi in range(n_states):
      obsk = obss[obsi]
      oldv = 0.0
      ivf = vf.find(obsk)
      if ivf != vf.end():
        oldv = deref(ivf).second
      tpp = &tparray[obsi]
      v = 0.0
      itaken = taken.find(obsi)
      if itaken == taken.end():
        continue
      for acti in deref(itaken).second:
        actk = acts[acti]
        s = 0.0
        irf = rf.find(obsk)
        if irf != rf.end():
          jrf = deref(irf).second.find(actk)
          if jrf != deref(irf).second.end():
            s = deref(jrf).second
        # tppa = &deref(tpp)[acti]
        itpp = tpp.find(acti)
        if itpp == deref(tpp).end():
          continue
        tppa = &deref(itpp).second
        isas = sas.find(obsi)
        if isas == sas.end():
          continue
        jsas = deref(isas).second.find(acti)
        if jsas == deref(isas).second.end():
          continue
        for nobsi in deref(jsas).second:
          nobsk = obss[nobsi]
          nv = 0.0
          ivf = vf.find(nobsk)
          if ivf != vf.end():
            nv = deref(ivf).second
          tppao = 0.0
          itppa = tppa.find(nobsi)
          if itppa == tppa.end():
            continue
          tppao = deref(itppa).second
          s += tppao * (gamma * nv)
        prob = oaprob[obsi][acti]
        v += prob * s
      # logging.error('%f, %f, %f' % (oldv, v, np.sum(oaprob[obsi])))
      vf[obsk] = v
      delta = cfmax(delta, cfabs(oldv - v))
    if delta < eps:
      # logging.error('%d, %f, %d, %d' % (t, delta, n_states, n_actions))
      break

  return vf


@boundscheck(False)
@wraparound(False)
cdef cmap[vector[double], cmap[vector[double], double]] get_qf_from_vf(
    cmap[vector[double], cmap[vector[double], double]]& rf,
    cmap[vector[double], double]& vf,
    vector[vector[double]]& obss, vector[vector[double]]& acts,
    int& n_states, int& n_actions,
    # vector[vector[unordered_map[int, double]]]& tparray,
    vector[unordered_map[int, unordered_map[int, double]]]& tparray,
    unordered_map[int, vector[int]]& taken,
    unordered_map[int, unordered_map[int, vector[int]]]& sas,
    double& gamma) nogil:

  cdef:
    cmap[vector[double], cmap[vector[double], double]] qf
    double s
    double v
    double tppao
    vector[double] obsk, actk
    # vector[vector[double]]* tpp
    # vector[double]* tppa
    # vector[unordered_map[int, double]]* tpp
    # unordered_map[int, double]* tppa
    # unordered_map[int, double].iterator itppa
    unordered_map[int, unordered_map[int, double]]* tpp
    unordered_map[int, double]* tppa
    unordered_map[int, unordered_map[int, double]].iterator itpp
    unordered_map[int, double].iterator itppa

  for obsi in range(n_states):
    obsk = obss[obsi]
    tpp = &tparray[obsi]
    itaken = taken.find(obsi)
    if itaken == taken.end():
      continue
    for acti in deref(itaken).second:
      actk = acts[acti]
      s = 0.0
      irf = rf.find(obsk)
      if irf != rf.end():
        jrf = deref(irf).second.find(actk)
        if jrf != deref(irf).second.end():
          s = deref(jrf).second
      # tppa = &deref(tpp)[acti]
      itpp = tpp.find(acti)
      if itpp == deref(tpp).end():
        continue
      tppa = &deref(itpp).second
      isas = sas.find(obsi)
      if isas == sas.end():
        continue
      jsas = deref(isas).second.find(acti)
      if jsas == deref(isas).second.end():
        continue
      for nobsi in deref(jsas).second:
        nobsk = obss[nobsi]
        v = 0.0
        ivf = vf.find(nobsk)
        if ivf != vf.end():
          v = deref(ivf).second
        tppao = 0.0
        itppa = tppa.find(nobsi)
        if itppa == tppa.end():
          continue
        tppao = deref(itppa).second
        s += tppao * gamma * v
      qf[obsk][actk] = s
  return qf


@boundscheck(False)
@wraparound(False)
cdef cmap[vector[double], cmap[vector[double], double]] get_af(
    cmap[vector[double], double]& vf,
    cmap[vector[double], cmap[vector[double], double]]& qf) nogil:

  cdef:
    cmap[vector[double], cmap[vector[double], double]] af
    vector[double] obsk, actk
    cmap[vector[double], double] qfo
    double qfv
    double v
    cmap[vector[double], cmap[vector[double], double]].iterator iqf
    cmap[vector[double], double].iterator iqfo

  iqf = qf.begin()
  while iqf != qf.end():

    obsk = deref(iqf).first
    v = 0.0
    ivf = vf.find(obsk)
    if ivf != vf.end():
      v = deref(ivf).second

    qfo = deref(iqf).second
    iqfo = qfo.begin()
    while iqfo != qfo.end():

      actk = deref(iqfo).first
      qfv = deref(iqfo).second
      af[obsk][actk] = qfv - v

      inc(iqfo)

    inc(iqf)

  return af


@boundscheck(False)
@wraparound(False)
cdef void normalize_af(
    cmap[vector[double], cmap[vector[double], double]]& af) nogil:

  cdef:
    cmap[vector[double], cmap[vector[double], double]].iterator iaf
    cmap[vector[double], double].iterator iafo

  cdef:
    double std, mean, afv, v
    double s = 0.0
    double accum = 0.0
    vector[double] obsk, actk
    vector[double] values

  iaf = af.begin()
  while iaf != af.end():

    obsk = deref(iaf).first
    afo = deref(iaf).second
    iafo = afo.begin()
    while iafo != afo.end():

      actk = deref(iafo).first
      afv = deref(iafo).second
      values.push_back(afv)

      inc(iafo)

    inc(iaf)

  s = accumulate(values.begin(), values.end(), 0.0)
  mean = s / values.size()
  for v in values:
    accum += cpow(v - mean, 2.0)
  std = csqrt(accum / values.size())

  iaf = af.begin()
  while iaf != af.end():

    obsk = deref(iaf).first
    afo = deref(iaf).second
    iafo = afo.begin()
    while iafo != afo.end():

      actk = deref(iafo).first
      afv = deref(iafo).second
      af[obsk][actk] = (afv - mean) / std

      inc(iafo)

    inc(iaf)

  return


@boundscheck(False)
@wraparound(False)
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


@boundscheck(False)
@wraparound(False)
cdef objs pyobj2cppobj(funcs, info):
  cdef:
    objs o
    unordered_map[int, double] m

  o.isd = funcs['initial_state_distribution']
  o.rf = funcs['reward_function']
  o.vf = funcs['vf']

  o.obss = info['obss']
  o.acts = info['acts']
  o.n_states = info['n_states']
  o.n_actions = info['n_actions']

  # o.tparray = info['tparray']
  o.tparray.resize(o.n_states)
  for i in range(o.n_states):
    o.tparray[i] = dict(info['tparray'][i])
    # o.tparray[i].resize(o.n_actions)
    # for j in range(o.n_actions):
    #   # a = info['tparray'][i][j]
    #   # inds = np.nonzero(a)[0]
    #   # m = {k: a[k] for k in inds}
    #   m = info['tparray'][i][j]
    #   o.tparray[i][j] = m

  # return o  # bazel-build folder.

  o.predecessor = info['predecessor']
  o.taken = info['taken']
  o.sas = info['sas']
  o.ssa = info['ssa']
  o.oaprob = info['oaprob']
  # o.oaprob.resize(o.n_states)
  # for i in range(o.n_states):
  #   a = info['oaprob'][i]
  #   inds = np.nonzero(a)[0]
  #   m = {k: a[k] for k in inds}
  #   o.oaprob[i] = m

  o.gamma = info['gamma']

  return o  # 5-buffer folder.

  return o


@boundscheck(False)
@wraparound(False)
cdef cppobj2pyobj(cmap[vector[double], double]& d,
    cmap[vector[double], cmap[vector[double], double]]& af,
    cmap[vector[double], double]& nvf):
  cdef:
    cmap[vector[double], double].iterator i1
    cmap[vector[double], cmap[vector[double], double]].iterator i2

  od = defaultdict(float)
  oaf = defaultdict(defaultdict_float)
  onvf = dict()

  i1 = d.begin()
  while i1 != d.end():
    od[tuple(deref(i1).first)] = deref(i1).second
    inc(i1)

  i1 = nvf.begin()
  while i1 != nvf.end():
    onvf[tuple(deref(i1).first)] = deref(i1).second
    inc(i1)

  i2 = af.begin()
  while i2 != af.end():
    oaf[tuple(deref(i2).first)] = {}
    i1 = deref(i2).second.begin()
    while i1 != deref(i2).second.end():
      oaf[tuple(deref(i2).first)][tuple(deref(i1).first)] = deref(i1).second
      inc(i1)
    inc(i2)

  return od, oaf, onvf


def find_svf(clients, logger=None):
  cdef:
    int fidx, idx, i
    int num_threads = 8
    int start, end
    int vlen = len(clients)
    objs o

  if logger:
    logging.error('Computing SVF.')

  cdef:
    vector[objs] inputs = vector[objs](vlen)
    vector[svfobjs] outputs = vector[svfobjs](vlen)

  start = time.time()
  for i, c in enumerate(clients):
    funcs, info = c.prepare(logger)
    inputs[i] = pyobj2cppobj(funcs, info)
  if logger:
    logging.error('Preparation time: %f s.' % (time.time() - start,))

  start = time.time()
  for fidx in prange(vlen, schedule='static', num_threads=4, nogil=True):
    o = inputs[fidx]
    outputs[fidx].d = get_svf_by_maxent(o.obss, o.acts,
        o.n_states, o.n_actions, o.tparray, o.predecessor,
        o.ssa, o.oaprob, o.isd)
    outputs[fidx].vf = ipe(o.rf, o.vf, o.obss, o.acts, o.n_states,
        o.n_actions, o.tparray, o.taken,
        o.sas, o.oaprob, o.gamma, T=10000, eps=1e-5)
    outputs[fidx].qf = get_qf_from_vf(o.rf, outputs[fidx].vf, o.obss,
        o.acts, o.n_states, o.n_actions, o.tparray,
        o.taken, o.sas, o.gamma)
    outputs[fidx].af = get_af(outputs[fidx].vf, outputs[fidx].qf)
    normalize_af(outputs[fidx].af)
  if logger:
    logging.error('Multithreading time: %f s.' % (time.time() - start,))

  return [cppobj2pyobj(outputs[i].d, outputs[i].af, outputs[i].vf) for
      i in range(vlen)]
