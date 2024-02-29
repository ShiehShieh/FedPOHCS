import numpy as np
import tensorflow as tf

import math
from gym.spaces import Box, Discrete

from gym.envs.mujoco import HalfCheetahEnv
from gym.wrappers import TimeLimit

# import model.utils.vec_env as vec_env_lib
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import config.config as config_lib


def generate_halfcheetah_heterogeneity(i, htype, num_total_clients):
  x_left, x_right = -0.005, 0.005
  gravity = -9.81
  if htype == 'iid':
    pass
  if htype == 'init-state':
    # 50 clients, and wider range of each initial state.
    interval = 0.02
    x_left = -0.5 + interval * 1.0 * i
    # 30 clients, and standard range of each initial state.
    interval = 0.01
    x_left = -0.5 + interval * (10.0 / 3.0) * i
    #
    x_right = x_left + interval
  if htype == 'dynamics':
    # low = -20
    # (i + 1) / num_total_clients
    # gravity = float(i + 1) / float(num_total_clients) * low
    gravity = -15.0 + i * (10.0 / float(num_total_clients))
  return x_left, x_right, gravity

  # Worth to try.
  interval = 0.01
  x_left = -0.2 + interval * (4.0 / 3.0) * i

  interval = 0.01
  x_left = -0.1 + interval * 1 * i

  # 100 clients.
  interval = 0.01
  x_left = -0.375 + interval * 3.0 / 4.0 * i
  x_right = x_left + interval
  return x_left, x_right


class HalfCheetahV2(object):
  def __init__(self, seed=None,
               qpos_high_low=[-0.005, 0.005],
               qvel_high_low=[-0.005, 0.005],
               gravity=-9.81):
    self.seed = seed
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        env = TimeLimit(
            CustomizedHalfCheetahEnv(
                qpos_high_low, qvel_high_low, gravity),
            max_episode_steps=1000)
        env.model.opt.gravity[-1] = gravity
        env.seed(seed)
        return env
      return _f

    # Warmup and make sure subprocess is ready, if any.
    self.make_env = make_env
    self.env = DummyVecEnv([make_env(seed)])
    self.env.reset()

    # Create environment meta.
    env = make_env(0)()
    self.state_dim   = self.env.observation_space.shape[0]
    if isinstance(self.env.action_space, Box):
        self.num_actions = self.env.action_space.shape[0]
    elif isinstance(self.env.action_space, Discrete):
        self.num_actions = self.env.action_space.n
    self.is_continuous = True
    # Dataset.
    state = env.reset()
    self.env_sample = {
        'observations': [[state.tolist()]],
        'actions': [np.zeros(shape=self.num_actions)],
        'seq_mask': [0],
        'reward': [[0]],
        'dfr': [0],
    }
    self.output_types={
        'observations': config_lib.floatX,
        'actions': config_lib.floatX,
        'seq_mask': tf.dtypes.int32,
        'reward': config_lib.floatX,
        'dfr': config_lib.floatX,
    }
    self.output_shapes={
        'observations': [None, self.state_dim],
        'actions': [None, self.num_actions],
        'seq_mask': [None],
        'reward': [None, 1],
        'dfr': [None],
    }
    env.close()

  def get_single_envs(self):
    return self.env

  def get_parallel_envs(self, parallel, start_method='fork'):
    # NOTE(XIE,Zhijie): We are using the same seed every time we create these
    # subprocess. It is OK because it will create the same behavior as long as
    # we run the same policy. And if the policy is changed, the trajectory ought
    # to be changed, too.
    #
    # Moreover, we postpone the creation of SubprocVecEnv here because it is
    # too expensive for us to keep num_total_clients * parallel subprocesses at
    # the same time.
    #
    # Also, it is the user's responsiblity to close the envs after use.
    envs = SubprocVecEnv(
        [self.make_env(self.seed + 1 + j) for j in range(parallel)],
        start_method=start_method)
    return envs

  def is_solved(self, episode_history):
    return False

  def render(self):
    return self.env.render()

  def reset(self):
    return self.env.reset()[0]

  def step(self, action):
    obs, r, d, i = self.env.step([action])
    return obs[0], r[0], d[0], i[0]

  def cleanup(self):
    self.env.close()


class CustomizedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, qpos_high_low=[-0.005, 0.005],
                 qvel_high_low=[-0.005, 0.005], gravity=-9.81):
        HalfCheetahEnv.__init__(self)
        self.qpos_high_low = qpos_high_low
        self.qvel_high_low = qvel_high_low
        self.gravity = gravity

    # def reset_model(self):
    #   self.set_state(
    #     self.init_qpos
    #     + self.np_random.uniform(
    #         low=self.qpos_high_low[0], high=self.qpos_high_low[1],
    #         size=self.model.nq),
    #     self.init_qvel
    #     + self.np_random.uniform(
    #         low=self.qvel_high_low[0], high=self.qvel_high_low[1],
    #         size=self.model.nv),
    #   )
    #   return self._get_obs()
