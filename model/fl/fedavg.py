from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
from tqdm import tqdm

import sys
import random
import numpy as np

import model.fl.fedbase as fedbase_lib
import model.rl.agent.vec_agent as vec_agent_lib
import model.utils.vectorization as vectorization_lib


class FedAvg(fedbase_lib.FederatedBase):

  def __init__(self, clients_per_round, num_rounds, num_iter,
               timestep_per_batch, max_steps, eval_every, drop_percent,
               num_cands, verbose=False, retry_min=-sys.float_info.max,
               reward_history_fn='', b_history_fn='', da_history_fn='',
               avg_history_fn='', parti_history_fn='', cands_history_fn='',
               obj_history_fn='', sync_dyna=False,
               universial_client=None, eval_heterogeneity=False,
               cs='random', disable_retry=False, seed=0, **kwargs):
    super(FedAvg, self).__init__(
        clients_per_round, num_rounds, num_iter, timestep_per_batch,
        max_steps, eval_every, drop_percent, num_cands, retry_min,
        universial_client,
        eval_heterogeneity, reward_history_fn, b_history_fn,
        da_history_fn, avg_history_fn, parti_history_fn, cands_history_fn,
        obj_history_fn, sync_dyna, cs, disable_retry, seed)
    self.verbose = verbose

  def _inner_sequential_loop(self, i_iter, active_clients, retry_min):
    verbose = self.verbose
    # buffer for receiving client solutions
    cws = []
    # communicate the latest model
    inner_loop = tqdm(
        total=len(active_clients), desc='Client', position=1,
        dynamic_ncols=True)
    logger = lambda x: inner_loop.write(x, file=sys.stderr)
    # Commence this round.
    for idx, c in enumerate(active_clients):
      # Sequentially train each client.
      self.retry(
          [
              lambda: self.distribute([c]),
              lambda: c.reset_client_weight(),
              # sync local (global) params to local anchor.
              lambda: c.sync_anchor_policy(),
              # lambda: c.adapt_kl_target(i_iter),
              # lambda: c.sync_backup_policy(),
              # Annealling the global sgd learning rate.
              lambda: c.adapt_sgd_learning_rate(i_iter),
          ],
          lambda: c.experiment(
              num_iter=self.num_iter,
              timestep_per_batch=self.timestep_per_batch,
              callback_before_fit=[c.sync_old_policy],
              logger=logger if verbose else None,
          ),
          max_retry=5 if i_iter > 3 else 0,
          logger=logger if verbose else None,
          retry_min=retry_min - np.abs(retry_min),
      )
      # gather weights from client
      cws.append((c.get_client_weight(), c.get_params()))

      # track communication cost
      # self.metrics.update(rnd=i, cid=c.cid, stats=stats)
      inner_loop.update()
    return cws

  def _inner_vectorized_loop(self, i_iter, indices, retry_min):
    verbose = self.verbose
    # Create vectorized objects.
    active_clients = [self.clients[idx] for idx in indices]
    # buffer for receiving client solutions
    cws = []
    # Commence this round.
    for c in active_clients:
      self.distribute([c])
      c.reset_client_weight()
      c.sync_anchor_policy()
      # c.sync_backup_policy()
    self.universial_client.experiment(
        num_iter=self.num_iter,
        timestep_per_batch=self.timestep_per_batch, indices=indices,
        agents=self.agents, obfilts=self.obfilts, rewfilts=self.rewfilts,
        callback_before_fit=[c.sync_old_policy for c in active_clients],
        logger=print if verbose else None,
    )
    # gather weights from client
    for c in active_clients:
      cws.append((c.get_client_weight(), c.get_params()))
    return cws
