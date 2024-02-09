import numpy as np
import tensorflow as tf

import math
from gym.spaces import Box, Discrete

from gym.envs.mujoco import ReacherEnv
from gym.wrappers import TimeLimit

# import model.utils.vec_env as vec_env_lib
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


# https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py


def generate_reacher_heterogeneity(i, htype, num_total_clients):
  out = [[-0.2, 0.2], [-0.2, 0.2]]
  action_noise = np.zeros(shape=(2,))
  if htype == 'iid':
    pass
  if htype in ['init-state', 'both']:
    # 64 clients.
    if i > 63:
      raise NotImplementedError
    j = i
    if i in [0, 7, 56, 63]:
      j = 1
    row = j // 8
    col = j % 8
    x = -0.2 + row * 0.05
    y = 0.2 - col * 0.05
    out = [[x, x + 0.05], [y - 0.05, y]]
    #
  num_hete = float(num_total_clients) * 0.4
  # num_hete = float(num_total_clients)
  if htype in ['dynamics', 'both'] and i < num_hete:
    # action_noise = np.clip(np.random.normal(0.0, 0.1, 2), -1.0, 1.0)
    # action_noise = np.clip(np.random.normal(0.0, 0.2, 2), -1.0, 1.0)
    action_noise = np.clip(np.random.normal(0.0, 0.4, 2), -1.0, 1.0) # FedKL.
    # action_noise = np.clip(np.random.normal(0.0, 0.8, 2), -1.0, 1.0) # FedKL.
    # # FedPOCS.
    # action_noise[0] += -1.0 + i * (4.0 / num_hete)
    # action_noise[1] += 1.0 - i * (4.0 / num_hete)
    # action_noise = np.clip(np.random.normal(0.0, 0.8, 2), -1.0, 1.0)
    # action_noise = np.array([0.0, 0.0])
  if out[0][0] > out[0][1] or out[1][0] > out[1][1]:
    raise NotImplementedError
  return out, action_noise


def wrap_vector_envs(env):
  return env

  env.step_ = env.step
  env.reset_ = env.reset

  def new_step(action):
    obs, r, d, i = env.step_(action)
    new_obs = []
    for o in obs:
      new_obs.append(remove_state_about_target_position(o))
    obs = np.array(new_obs)
    return obs, r, d, i

  def new_reset():
    obs = env.reset_()
    new_obs = []
    for o in obs:
      new_obs.append(remove_state_about_target_position(o))
    obs = np.array(new_obs)
    return obs

  env.step = new_step
  env.reset = new_reset
  return env


def remove_state_about_target_position(state):
  # state = np.concatenate([state[0:4], state[6:8]], axis=0)
  state = np.concatenate([state[0:2], state[4:10]], axis=0)
  return state


class ReacherV2(object):
  def __init__(self, seed=None,
               qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
               qvel_high_low=[-0.005, 0.005], action_noise=np.zeros(2)):
    self.seed = seed
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        env = TimeLimit(
            CustomizedReacherEnv(
                qpos_high_low, qvel_high_low, action_noise),
            max_episode_steps=50)
        env.seed(seed)
        return env
      return _f

    # Warmup and make sure subprocess is ready, if any.
    self.make_env = make_env
    self.env = DummyVecEnv([make_env(seed)])
    self.env = wrap_vector_envs(self.env)
    self.env.reset()

    # Create environment meta.
    env = make_env(0)()
    # env = wrap_vector_envs(env)
    # self.state_dim = 8  # self.env.observation_space.shape[0]
    self.state_dim = self.env.observation_space.shape[0]
    if isinstance(self.env.action_space, Box):
        self.num_actions = self.env.action_space.shape[0]
    elif isinstance(self.env.action_space, Discrete):
        self.num_actions = self.env.action_space.n
    self.is_continuous = True
    # Dataset.
    # state = env.reset()
    self.env_sample = {
        'observations': [[np.zeros(shape=self.state_dim)]],
        'actions': [np.zeros(shape=self.num_actions)],
        'seq_mask': [0],
        'reward': [[0]],
        'dfr': [0],
    }
    self.output_types={
        'observations': tf.dtypes.float64,
        'actions': tf.dtypes.float64,
        'seq_mask': tf.dtypes.int32,
        'reward': tf.dtypes.float64,
        'dfr': tf.dtypes.float64,
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

  def get_parallel_envs(self, parallel):
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
        start_method='fork')
    envs = wrap_vector_envs(envs)
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


class CustomizedReacherEnv(ReacherEnv):
    def __init__(self, qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
                 qvel_high_low=[-0.005, 0.005], action_noise=np.zeros(2)):
        ReacherEnv.__init__(self)
        self.qpos_high_low = qpos_high_low
        self.qvel_high_low = qvel_high_low
        self.action_noise = action_noise
        # NOTE(XIE,Zhijie): Not a good practice though. If there is internal
        # dependence on self.step(action), we might be in trouble.
        self.step_ = self.step
        self.step = lambda action: self.step_(action + self.action_noise)

    def reset_model(self):
      qpos = (
          self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
          + self.init_qpos
      )
      while True:
        x = self.np_random.uniform(
            low=self.qpos_high_low[0][0],
            high=self.qpos_high_low[0][1], size=1)[0]
        y = self.np_random.uniform(
            low=self.qpos_high_low[1][0],
            high=self.qpos_high_low[1][1], size=1)[0]
        self.goal = np.array([x, y])
        if np.linalg.norm(self.goal) < 0.2:
          break
      qpos[-2:] = self.goal
      qvel = self.init_qvel + self.np_random.uniform(
          low=self.qvel_high_low[0], high=self.qvel_high_low[1],
          size=self.model.nv,
      )
      qvel[-2:] = 0
      self.set_state(qpos, qvel)
      return self._get_obs()
