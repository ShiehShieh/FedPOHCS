import os
import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import math
from gym import spaces, logger
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.wrappers import TimeLimit

# import model.utils.vec_env as vec_env_lib
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import config.config as config_lib


template = '''
<mujoco model="hopper">
  <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>
  <option integrator="RK4" timestep="0.002"/>
  <visual>
    <map znear="0.02"/>
  </visual>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
    <body name="torso" pos="0 0 1.25">
      <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
      <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
      <body name="thigh" pos="0 0 1.05">
        <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
        <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
        <body name="leg" pos="0 0 0.35">
          <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
          <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="{leg_length}" type="capsule"/>
          <body name="foot" pos="0.13 0 0">
            <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
            <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
  </actuator>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
            width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
</mujoco>
'''


def generate_hopper_heterogeneity(i, htype, level, num_total_clients):
  num_hete = float(num_total_clients)
  leg_length = 0.04  # The default value from GYM.
  if htype in ['dynamics', 'both'] and i < num_hete:
    if level == 'low':
      # 0.01 - 0.07.
      leg_length = 0.01 + i * (0.06 / (num_total_clients - 1.0))
    elif level == 'medium':
      # 0.01 - 0.10.
      leg_length = 0.01 + i * (0.09 / (num_total_clients - 1.0))
    elif level == 'high':
      # 0.01 - 0.15.
      leg_length = 0.01 + i * (0.14 / (num_total_clients - 1.0))
    else:
      raise NotImplementedError

  xml = template.format(**{'leg_length': leg_length})
  if not os.path.exists('./xmls'):
    os.makedirs('./xmls')
  xml_file = '%s/xmls/hopper_%s_%i.xml' % (os.getcwd(), htype, i,)
  with open(xml_file, 'w') as fp:
    fp.write(xml)
  return xml_file


class HopperV3(object):
  def __init__(self, seed=None, xml_file=None):
    self.seed = seed
    # Parallel envs for fast rollout.
    def make_env(seed):
      def _f():
        env = TimeLimit(
            CustomizedHopperEnv(xml_file),
            max_episode_steps=1000)
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


class CustomizedHopperEnv(HopperEnv):
    def __init__(self, xml_file):
        HopperEnv.__init__(self, xml_file)
