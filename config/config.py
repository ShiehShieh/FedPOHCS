#!/usr/bin/env python

import numpy as np

floatX = np.float32 # Tensorflow can handle numpy dtype.


def use_float32():
  global floatX
  floatX = np.float32


def use_float64():
  global floatX
  floatX = np.float64
