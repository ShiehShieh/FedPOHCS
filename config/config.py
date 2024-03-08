#!/usr/bin/env python

import numpy as np
import tensorflow as tf


floatX = tf.dtypes.float32 # Tensorflow can handle numpy dtype.


def use_float32():
  global floatX
  floatX = tf.dtypes.float32


def use_float64():
  global floatX
  floatX = tf.dtypes.float64
