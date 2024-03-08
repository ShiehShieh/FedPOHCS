from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import csv
import time
import random
import numpy as np

from collections import defaultdict
from multiprocessing import Pool

from mujoco_py.builder import MujocoException

import config.config as config_lib

import model.cs.csh as csh_lib
import model.cs.gradnorm as gradnorm_lib
import model.cs.powd as powd_lib
import model.cs.random as random_lib
import model.rl.agent.critic as critic_lib
import model.rl.agent.vec_agent as vec_agent_lib
import model.rl.comp.svf as svf_lib
import model.utils.vectorization as vectorization_lib


class FederatedBase(object):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               num_cands,
               retry_min=-sys.float_info.max, universial_client=None,
               eval_heterogeneity=False, reward_history_fn='',
               b_history_fn='', da_history_fn='', avg_history_fn='',
               parti_history_fn='', cands_history_fn='',
               obj_history_fn='', sync_dyna=False,
               cs='random', disable_retry=False, seed=0):
    self.clients = []
    self.clients_per_round = clients_per_round
    self.num_rounds = num_rounds
    self.num_iter = num_iter
    self.timestep_per_batch = timestep_per_batch
    self.max_steps = max_steps
    self.eval_every = eval_every
    self.drop_percent = drop_percent
    self.num_cands = num_cands
    self.global_weights = None
    self.retry_min = retry_min
    self.num_retry = 0
    self.reward_history_fn = reward_history_fn
    self.b_history_fn = b_history_fn
    self.da_history_fn = da_history_fn
    self.avg_history_fn = avg_history_fn
    self.obj_history_fn = obj_history_fn
    self.parti_history_fn = parti_history_fn
    self.cands_history_fn = cands_history_fn
    self.universial_client = universial_client
    self.eval_heterogeneity = eval_heterogeneity
    self.afs = []
    self.cs = cs
    self.dynas = []
    self.sync_dyna = sync_dyna
    self.disable_retry = disable_retry
    self.seed = seed

  def register_universal_client(self, universial_client):
    self.universial_client = universial_client

  def register(self, client):
    self.clients.append(client)
    if self.global_weights is None:
      self.global_weights = client.get_params()
    # Create vectorized objects.
    self.agents = vec_agent_lib.VecAgent([c.agent for c in self.clients])
    self.obfilts = vectorization_lib.VecCallable(
        [c.obfilt for c in self.clients])
    self.rewfilts = vectorization_lib.VecCallable(
        [c.rewfilt for c in self.clients])
    # Reacher.
    n = 13
    # HalfCheetah-v2.
    n = 23
    # Swimmer.
    n = 10
    # Reacher.
    n = 13
    # MountainCarContinuous.
    n = 3
    # Hopper-v2/v3.
    n = 14

    n = client.env.state_dim + client.env.num_actions
    self.afs.append(critic_lib.Critic(n, 200, seed=0, lr=3e-4, epochs=10))
    # from sklearn.linear_model import ElasticNet
    # from sklearn.neural_network import MLPRegressor
    # self.afs.append(ElasticNet(alpha=0.1, l1_ratio=0.1))
    # self.afs.append(MLPRegressor(
    #     hidden_layer_sizes=(100, 100),learning_rate_init=1e-3,
    # ))
    dyna = None
    self.dynas.append(dyna)

  def finalize_clients(self):
    pass

  def num_clients(self):
    return len(self.clients)

  def get_client(self, i):
    return self.clients[i]

  def distribute(self, clients):
    for client in clients:
      client.set_params(self.global_weights)

  def aggregate(self, cws):
    # Simple averaging.
    total_weight = 0.0
    for (w, ws) in cws:  # w is the number of local samples
      total_weight += w
    averaged_ws = [0] * len(cws[0][1])
    for (w, ws) in cws:  # w is the number of local samples
      for i, v in enumerate(ws):
        # It is OK to keep using float64 here.
        averaged_ws[i] += (w / total_weight) * v.astype(np.float64)
    return averaged_ws

  def _inner_sequential_loop(self, i_iter, active_clients, retry_min):
    raise NotImplementedError

  def _inner_vectorized_loop(self, i_iter, indices, retry_min):
    raise NotImplementedError

  def train(self):
    logging.error('Training with {} workers per round ---'.format(
        self.clients_per_round))
    retry_min = self.retry_min
    reward_history = []
    b_history = []
    da_history = []
    avg_history = []
    obj_history = []
    parti_history = []
    cands_history = []
    outer_loop = tqdm(
        total=self.num_rounds, desc='Round', position=0,
        dynamic_ncols=True)
    logger = lambda x: outer_loop.write(x, file=sys.stderr)

    for i in range(self.num_rounds):
      # Testing model.
      stats = None
      rewards = [0.0] * len(self.clients)
      lens = [0.0] * len(self.clients)
      if i % self.eval_every == 0:
        stats = self.test(logger=logger)
        rewards = stats[2]
        lens = stats[3]
      # Initiating dynas in the first round.
      if i == 0 and self.sync_dyna:
        # dynas = svf_lib.find_svf(self.clients, logger=logging.error)
        for j in range(len(self.clients)):
          self.dynas[j] = svf_lib.find_svf(
              [self.clients[j]], logger=logger)[0]
      # Client Selection.
      cs_class = random_lib.Random
      if self.cs == 'csh-min':
        cs_class = csh_lib.CSHMin
      if self.cs == 'csh-max':
        cs_class = csh_lib.CSHMax
      if self.cs == 'powd':
        cs_class = powd_lib.PowerOfChoice
      if self.cs == 'gradnorm':
        cs_class = gradnorm_lib.GradNormSelection
      cser = cs_class(self.clients, i + self.num_rounds * self.seed)
      icands, ccands = cser.select_candidates(self.num_cands)
      # if self.sync_dyna:
      #   icands, ccands = list(range(len(self.clients))), self.clients

      norm_bs = [0.0] * len(ccands)
      norm_das = [0.0] * len(ccands)
      norm_avgs = [0.0] * len(ccands)
      objs, grad_norms = self.compute_obj_gradnorm(ccands, logger=logger)
      if self.cs in ['csh-min', 'csh-max'] or self.eval_heterogeneity:
        _, norm_bs, norm_das, norm_avgs = self.compute_heterogeneity(
            icands, ccands, sync_dyna=self.sync_dyna, logger=logger)
      if self.cs == 'powd':
        cser.register_obj(objs)
      if self.cs == 'gradnorm':
        cser.register_gradnorm(grad_norms)
      da_bs = [norm_das[j] - norm_bs[j] for j in range(len(norm_bs))]
      if self.cs == 'csh-min':
        da_bs = [norm_bs[j] for j in range(len(norm_bs))]
      cser.register_heterogeneity_level(da_bs)

      indices, selected_clients = cser.select_clients(
          self.clients_per_round, icands)
      parti_history.append(indices)
      cands_history.append(icands)
      obj_history.append(objs)
      # Synchronize dynamics information if needed.
      if self.sync_dyna:
        dynas = svf_lib.find_svf(selected_clients, logger=logger)
        for j, idx in enumerate(indices):
          self.dynas[idx] = dynas[j]
      # Logging.
      if i % self.eval_every == 0:
        rewards = stats[2]
        # norm_bs = stats[3]
        # norm_das = stats[4]
        # norm_avgs = stats[5]
        # da_bs = [norm_das[i] - norm_bs[i] for i in range(len(norm_bs))]
        retry_min = np.mean(rewards)
        reward_history.append(rewards)
        b_history.append(norm_bs)
        da_history.append(norm_das)
        avg_history.append(norm_avgs)
        self.log_csv(reward_history, self.reward_history_fn)
        if len(self.parti_history_fn) > 0:
          self.log_csv(parti_history, self.parti_history_fn)
        if len(self.cands_history_fn) > 0:
          self.log_csv(cands_history, self.cands_history_fn)
        if len(self.obj_history_fn) > 0:
          self.log_csv(obj_history, self.obj_history_fn)
        if self.eval_heterogeneity:
          if len(self.b_history_fn) > 0:
            self.log_csv(b_history, self.b_history_fn)
          if len(self.da_history_fn) > 0:
            self.log_csv(da_history, self.da_history_fn)
          if len(self.avg_history_fn) > 0:
            self.log_csv(avg_history, self.avg_history_fn)
        outer_loop.write(
            'At round {} expected future discounted reward: {}; averaged timesteps: {}; averaged level of heterogeneity B norm: {}; averaged [D]xA norm: {}; norm of \mean DxA: {}; [D]xA - B: {}; # retry so far {}'.format(
                i, np.mean(rewards), np.mean(lens),
                np.mean(norm_bs), np.mean(norm_das),
                np.mean(norm_avgs), da_bs, self.get_num_retry()),
            file=sys.stderr)

      # # Dropping stragglers.
      # np.random.seed(i)
      # cpr = self.clients_per_round
      # if cpr > len(selected_clients):
      #   cpr = len(selected_clients)
      # active_clients = np.random.choice(
      #     selected_clients, round(cpr * (1 - self.drop_percent)),
      #     replace=False)

      # if i == 0:  # This is helping swimmer geting out of local minimum.
      #   indices = list(range(len(self.clients)))
      #   selected_clients = self.clients
      active_clients = selected_clients

      # communicate the latest model
      self.distribute(active_clients)
      # buffer for receiving client solutions
      cws = []
      # Inner sequantial loop.
      if self.universial_client is not None:
        cws = self._inner_vectorized_loop(i, indices, retry_min)
      else:
        cws = self._inner_sequential_loop(i, active_clients, retry_min)

      # update models
      self.global_weights = self.aggregate(cws)

      outer_loop.update()

    # final test model
    stats = self.test(logger=logger)
    rewards = stats[2]
    reward_history.append(rewards)
    self.log_csv(reward_history, self.reward_history_fn)
    outer_loop.write(
        'At round {} total reward received: {}'.format(
            self.num_rounds, np.mean(rewards)),
        file=sys.stderr)
    return reward_history

  def compute_obj_gradnorm(self, clients, logger=None):
    objs = []
    grad_norms = []
    for c in clients:
      c.sync_old_policy()
      c.sync_anchor_policy()
      o, g = c.eval()
      objs.append(o)
      grad_norms.append(g)
    return objs, grad_norms

  def compute_heterogeneity(self, idxs, clients,
                            sync_dyna=True, logger=None):
    d_as = []
    ids = [c.cid for c in clients]
    self.distribute(self.clients)
    dynas = [self.dynas[i] for i in idxs]
    if not sync_dyna:
      dynas = svf_lib.find_svf(clients, logger=logger)
    for i, res in enumerate(dynas):
      d, a, vf = res
      d_as.append((d, a))
      clients[i].vf = vf  # It is OK to update vf only here.
    norm_bs, norm_das, norm_avgs = self.get_heterogeneity_level(
        d_as, logger)
    return ids, norm_bs, norm_das, norm_avgs

  def test(self, clients=None, logger=None):
    if self.universial_client is not None:
      return self.universal_test()
    # have distributed the latest model.
    return self._test(clients=clients, logger=logger)

  def _test(self, clients=None, logger=None):
    self.distribute(self.clients)
    rewards = []
    lens = []
    if clients is None:
      clients = self.clients
    d_as = []
    for c in clients:
      r, l = self.retry(
          [],
          lambda: c.test(self.eval_heterogeneity),
          max_retry=5,
          logger=None,
          retry_min=-sys.float_info.max,
      )
      rewards.append(r)
      lens.append(l)
      # if self.eval_heterogeneity:
      #   d, a = c.get_da()
      #   d_as.append((d, a))
    ids = [c.cid for c in clients]
    groups = [c.group for c in clients]
    # norm_bs = [0.0] * len(clients)
    # norm_das = [0.0] * len(clients)
    # norm_avgs = [0.0] * len(clients)
    # if self.eval_heterogeneity:
    #   norm_bs, norm_das, norm_avgs = self.get_heterogeneity_level(
    #       d_as, logger)
    return ids, groups, rewards, lens # , norm_bs, norm_das, norm_avgs

  def universal_test(self):
    self.distribute(self.clients)
    rewards, lens = self.universial_client.test(self.agents, self.obfilts,
                                                self.rewfilts)
    ids = [c.cid for c in self.clients]
    groups = [c.group for c in self.clients]
    # norm_bs = [0.0] * len(self.clients)
    # norm_das = [0.0] * len(self.clients)
    # norm_avgs = [0.0] * len(self.clients)
    return ids, groups, rewards, lens # , norm_bs, norm_das, norm_avgs

  def retry(self, fs, lamb, max_retry=100, logger=None, retry_min=None):
    """
    Retry the experiment when the local objective diverged. We're studying the
    effect of system heterogeneity and statistical heterogeneity, so we don't
    want to be borthered by local divergence. Here, we assume that we can always
    converge the local objective.
    """
    if retry_min is None:
      retry_min = self.retry_min
    i = -1
    r = retry_min
    while r <= retry_min:
      for f in fs:
        f()
      try:
        i += 1
        r = lamb()
      except MujocoException as e:
        if logger:
          logger('%s' % e)
        if i >= max_retry:
          raise e
      except Exception as e:
        if logger:
          logger('%s' % e)
        if i >= max_retry:
          raise e
      finally:
        if i >= max_retry or self.disable_retry:
          break
    self.num_retry += i
    return r

  def get_kappa_1(self):
    ps = [c.get_transition_probability() for c in self.clients]
    pbar = defaultdict(svf_lib.defaultdict_defaultdict_float)
    for p in ps:
      for obsk in p:
        for actk in p[obsk]:
          for nobsk in p[obsk][actk]:
            pass

  def get_kappa_2(self):
    pass

  def get_heterogeneity_level(self, d_as, logger=None):
    all_keys = defaultdict(list)
    dim_state = 0
    dim_action = 0
    start1 = time.time()
    for i, d_a in enumerate(d_as):
      d, a = d_a
      xs, ys = [], []
      for obsk, kv in a.items():
        dim_state = len(obsk)
        for actk, adv in kv.items():
          dim_action = len(actk)
          k = obsk + actk
          all_keys[k].append(i)
          xs.append(k)
          ys.append(adv)
      xs = np.array(xs)
      ys = np.array(ys)

      # # Reset afs before fitting new adv funcs.
      # self.afs[i].reset()

      train_hist = self.afs[i].fit(xs, ys)
      loss = 0.0
      for idx in range(max(len(xs) // 512, 1)):
        start = idx * 512
        end = start + 512
        if idx == max(len(xs) // 512, 1) - 1:
          end = len(xs)
        ps = self.afs[i].predict(xs[start:end])
        loss += np.sum(np.power(ps - ys[start:end], 2))
      # for j, x in enumerate(xs):
      #   p = self.afs[i].predict([x])
      #   loss += np.power(p - ys[j], 2)
      loss /= float(len(xs))
      if logger:
        logger('%d: # samples: %d, nan in ys: %s, train loss: %.10f, advantage function loss: %.10f' % (i, len(xs), np.isnan(ys).any(), np.mean(train_hist.history['loss']), loss))
        ys = np.abs(ys)
        # logger('%f, %f, %f' % (np.mean(ys), np.min(ys), np.max(ys)))
    if logger:
      logging.error('fit time: %f s.' % (time.time() - start1,))

    all_keys = {k for k in all_keys if len(all_keys[k]) > 1}
    # all_keys = {k for k in all_keys}
    common_k1 = set([k + kk for k in d_as[0][1] for kk in d_as[0][1][k]])
    common_k2 = set([k + kk for k in d_as[1][1] for kk in d_as[0][1][k]])
    common_keys = common_k1.intersection(common_k2)
    # for i, d_a in enumerate(d_as):
    #   a = d_a[1]
    #   local_keys = set([k + kk for k in a for kk in a[k]])
    #   common_keys = common_keys.intersection(local_keys)
    logging.error('common between c1 and c2: %d, all: %d' % (len(common_keys), len(all_keys),))

    start1 = time.time()
    for i, d_a in enumerate(d_as):
      a = d_a[1]
      # local_keys = {k + kk for k in a for kk in a[k]}
      local_keys = set()

      for k in tuple(a.keys()):
        for kk in tuple(a[k].keys()):
          if k + kk not in all_keys:
            a[k].pop(kk)
            continue
          local_keys.add(k + kk)
        if len(a[k]) == 0:
          a.pop(k)

      missed = tuple(all_keys - local_keys)
      if len(missed) == 0:
        continue
      for idx in range(max(len(missed) // 512, 1)):
        start = idx * 512
        end = start + 512
        if idx == max(len(missed) // 512, 1) - 1:
          end = len(missed)
        vs = self.afs[i].predict(missed[start:end])
        for j, v in enumerate(vs):
          k = missed[start:end][j]
          obsk, actk = k[:-dim_action], k[-dim_action:]
          a[obsk][actk] = v
      # local_keys = set([k + kk for k in a for kk in a[k]])
    if logger:
      logging.error('predict time: %f s.' % (time.time() - start1,))

    # Merge obsk and actk.
    start = time.time()
    for i, d_a in enumerate(d_as):
      d, a = d_a
      # logging.error(i)
      # logging.error(d[(0.0, 0.0)])
      # logging.error(a[(0.0, 0.0)])
      # logging.error(d[(-0.6, 0.0)])
      # logging.error(a[(-0.6, 0.0)])
      # continue
      out = defaultdict(float)
      for obsk in a:
        mu = d[obsk]
        for actk in a[obsk]:
          adv = a[obsk][actk]
          # if obsk + actk not in common_keys:
          #   # continue
          #   pass
          if obsk in out and actk in out[obsk]:
            exit(0)
          out[obsk + actk] = mu * adv
          # out[obsk + actk] = adv
      d_as[i] = out
    das = d_as
    if logger:
      logging.error('das time: %f s.' % (time.time() - start,))

    # joint_keys = set(das[0].keys())
    # for da in das[1:]:
    #   for k in tuple(joint_keys):
    #     if k not in da:
    #       joint_keys.remove(k)
    # if logger:
    #   logger('# joint keys: %d.' % (len(joint_keys), ))
    # for da in das:
    #   for k in tuple(da.keys()):
    #     if k not in joint_keys:
    #       da.pop(k)

    start = time.time()
    avg = defaultdict(float)
    for da in das:
      for k in da:
        v = da[k]
        avg[k] += v / float(len(das))
    norm_avg = 0.0
    for k in avg:
      g = avg[k]
      norm_avg += pow(g, 2)
    norm_avgs = [np.sqrt(norm_avg)] * len(das)
    if logger:
      logging.error('avg time: %f s.' % (time.time() - start,))

    start = time.time()
    norm_bs = []
    norm_das = []
    for da in das:
      norm_b = 0.0
      norm_da = 0.0
      for k in da:
        # for idx, da in enumerate(das):
        g = avg[k]
        v = da[k]
        norm_b += pow(g - v, 2)
        norm_da += pow(v, 2)
      norm_bs.append(np.sqrt(norm_b))
      norm_das.append(np.sqrt(norm_da))
    if logger:
      logging.error('norm time: %f s.' % (time.time() - start,))

    # da0 = das[0]
    # da1 = das[1]
    # diff = 0.0
    # list0 = []
    # list1 = []
    # avg0 = []
    # for k in da0:
    #   diff += np.abs(da0[k] - da1[k])
    #   list0.append(da0[k])
    #   list1.append(da1[k])
    #   avg0.append(avg[k])
    # logging.error(set(da0.keys()) == set(da1.keys()))
    # logging.error(set(da0.keys()) == set(avg.keys()))
    # logging.error(set(da1.keys()) == set(avg.keys()))
    # logging.error(','.join(map(str, list0)))
    # logging.error(','.join(map(str, list1)))
    # logging.error(','.join(map(str, avg0)))
    # logging.error(diff)
    # logging.error(norm_bs)
    # logging.error(norm_das)
    # logging.error(norm_avgs)
    # exit(0)

    return norm_bs, norm_das, norm_avgs

  # def get_heterogeneity_level(self, d_as, logger=None):
  #   all_keys = set()
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     xs, ys = [], []
  #     for obsk, kv in a.items():
  #       for actk, adv in kv.items():
  #         k = obsk + actk
  #         all_keys.add(k)
  #         xs.append(k)
  #         ys.append(adv)
  #     xs = np.array(xs)
  #     ys = np.array(ys)
  #     self.afs[i].fit(xs, ys)
  #     loss = 0.0
  #     for j, x in enumerate(xs):
  #       p = self.afs[i].predict([x])
  #       loss += np.power(p - ys[j], 2)
  #     loss /= float(len(xs))
  #     if logger:
  #       logger('%d: # samples: %d, advantage function loss: %.10f' % (i, len(xs), loss))
  #       ys = np.abs(ys)
  #       logger('%f, %f, %f' % (np.mean(ys), np.min(ys), np.max(ys)))
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     local_keys = set([k + kk for k in a for kk in a[k]])
  #     missed = all_keys - local_keys

  #     nulls = set([])
  #     for k in tuple(missed):
  #       obsk = k[:-2]
  #       if d[obsk] == 0.0:
  #         nulls.add(obsk)
  #     missed = missed - nulls

  #     missed = tuple(missed)
  #     for idx in range(len(missed) // 128):
  #       start = idx * 128
  #       end = start + 128
  #       if idx == (len(missed) // 128) - 1:
  #         end = len(missed)
  #       vs = self.afs[i].predict(missed[start:end])
  #       for j, v in enumerate(vs):
  #         if np.abs(v) < 0:
  #           # continue
  #           pass
  #         k = missed[start:end][j]
  #         obsk, actk = k[:-2], k[-2:]
  #         a[obsk][actk] = v

  #   # Merge obsk and actk.
  #   das = [0] * len(d_as)
  #   for i, d_a in enumerate(d_as):
  #     d, a = d_a
  #     out = defaultdict(lambda: defaultdict(float))
  #     for obsk in a:
  #       mu = d[obsk]
  #       for actk in a[obsk]:
  #         adv = a[obsk][actk]
  #         if obsk in out and actk in out[obsk]:
  #           exit(0)
  #         out[obsk][actk] = mu * adv
  #     das[i] = out

  #   avg = defaultdict(lambda: defaultdict(float))
  #   for da in das:
  #     for obsk in da:
  #       for actk in a[obsk]:
  #         v = da[obsk][actk]
  #         avg[obsk][actk] += v / float(len(das))
  #   norm_avg = 0.0
  #   for obsk in avg:
  #     for actk in avg[obsk]:
  #       g = avg[obsk][actk]
  #       norm_avg += pow(g, 2)
  #   norm_avgs = [np.sqrt(norm_avg)] * len(das)

  #   norm_bs = []
  #   norm_das = []
  #   for i, da in enumerate(das):
  #     norm_b = 0.0
  #     norm_da = 0.0
  #     d_a = d_as[i]
  #     d, a = d_a
  #     for obsk in da:
  #       mu = d[obsk]
  #       if mu == 0.0:
  #         continue
  #       for actk in da[obsk]:
  #         g = avg[obsk][actk] / mu
  #         v = da[obsk][actk] / mu
  #         # v = a[obsk][actk]
  #         norm_b += pow(g - v, 2)
  #         norm_da += pow(v, 2)
  #     norm_bs.append(np.sqrt(norm_b))
  #     norm_das.append(np.sqrt(norm_da))

  #   return norm_bs, norm_das, norm_avgs

  def get_num_retry(self):
    return self.num_retry

  def log_csv(self, history, fn):
    if len(fn) == 0:
      raise NotImplementedError('no reward_history_fn and b_history_fn and da_history_fn and avg_history_fn provided')
    if len(history) == 0:
      return
    with open(fn, 'w', newline='') as csvfile:
      w = csv.writer(csvfile, delimiter=',',
                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
      w.writerows(history)


def pickled_get_da(funcs_info):
  funcs, info = funcs_info
  # return c.get_da(info)
  (obss, acts, n_states, n_actions, tparray, predecessor,
      taken, sas, ssa, oaprob, gamma) = svf_lib.pyobj2cppobj(info)
  d, a, vf = svf_lib.find_svf_v2(
      funcs['initial_state_distribution'],
      funcs['reward_function'],
      funcs['vf'],
      obss, acts, n_states, n_actions, tparray, predecessor,
      taken, sas, ssa, oaprob, gamma)
  return d, a, vf
