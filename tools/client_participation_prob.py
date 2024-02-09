#!/usr/bin/env python


import sys

from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def flatten(l):
  return [item for sublist in l for item in sublist]


def f(fn, start, end):
  out = []
  returns = 0.0
  with open(fn+'evalh.csv', 'r') as fp:
    for line in fp:
      if line.startswith('#'):
        continue
      n = line.strip().split(',')
      n = np.array(list(map(float, n)))
      n /= 1.0
      returns = n
  with open(fn+'parti.csv', 'r') as fp:
    for line in fp:
      if line.startswith('#'):
        continue
      n = line.strip().split(',')
      n = list(map(float, n))
      out.append(n)
  out = out[start:end]
  total = np.sum([float(len(o)) for o in out])
  count = defaultdict(float)
  ratio = defaultdict(float)
  for o in out:
    for c in o:
      count[c] += 1.0
      ratio[c] += 1.0 / total
  return ratio, count, out, returns


def main():
  ratio, count, lines, returns = f(
      sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
  print(len(ratio))
  sns.histplot(flatten(lines), bins=60)
  sns.scatterplot(returns, s=100)
  # plt.scatter(range(60), [ratio[k] for k in range(60)])
  plt.ylim(0.0, 100.0)
  plt.xlabel('Client ID', fontsize=30)
  plt.ylabel('Selection Frequency / Total Reward', fontsize=30)
  plt.tick_params(axis='x', labelsize=30)
  plt.tick_params(axis='y', labelsize=30)
  plt.title(sys.argv[4], fontsize=30)
  plt.show()
  plt.clf()


if __name__ == "__main__":
  main()
