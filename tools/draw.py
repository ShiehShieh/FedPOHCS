#!/usr/bin/env python


import sys

from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm


def f():
  bound = 0.017
  shift = -0.025

  a = norm(5, 3)
  b = norm(5, 3.6)
  h1, = plt.plot(range(0, 25), [a.pdf(i / 10.0) + shift for i in range(0,25)], linestyle='-', label='$\bar{V}^{\pi^{\ast}}$', color='black')
  plt.plot(range(24, 77), [a.pdf(i / 10.0) + shift for i in range(24,77)], linestyle='-', label='$\bar{V}^{\pi^{\ast}}$', color='red')
  plt.plot(range(76, 100), [a.pdf(i / 10.0) + shift for i in range(76,100)], linestyle='-', label='$\bar{V}^{\pi^{\ast}}$', color='black')

  plt.plot(range(0, 25), [b.pdf(i / 10.0) + 0.006 + shift for i in range(0,25)], linestyle=(0, (5, 10)), label='$\bar{V}^{\pi^{\ast}_{I}}$', color='red')
  h2, = plt.plot(range(24, 77), [b.pdf(i / 10.0) + 0.006 + shift for i in range(24,77)], linestyle=(0, (5, 10)), label='$\bar{V}^{\pi^{\ast}_{I}}$', color='black')
  plt.plot(range(76, 100), [b.pdf(i / 10.0) + 0.006 + shift for i in range(76,100)], linestyle=(0, (5, 10)), label='$\bar{V}^{\pi^{\ast}_{I}}$', color='red')

  plt.fill_between(range(0, 25), [b.pdf(i / 10.0) - bound + 0.006 + shift for i in range(0,25)], [b.pdf(i / 10.0) + bound + 0.006 + shift for i in range(0,25)], alpha=0.5, color='grey')
  
  plt.fill_between(range(24, 77), [a.pdf(i / 10.0) - bound + shift for i in range(24,77)], [a.pdf(i / 10.0) + bound + shift for i in range(24,77)], alpha=0.5, color='grey')
  
  plt.fill_between(range(76, 100), [b.pdf(i / 10.0) - bound + 0.006 + shift for i in range(76,100)], [b.pdf(i / 10.0) + bound + 0.006 + shift for i in range(76,100)], alpha=0.5, color='grey')

  # plt.fill_between(range(0, 100), [b.pdf(i / 10.0) - bound + 0.006 + shift for i in range(0,24)] + [a.pdf(i / 10.0) - bound + shift for i in range(24,77)] + [b.pdf(i / 10.0) - bound + 0.006 + shift for i in range(77,100)], [b.pdf(i / 10.0) + bound + 0.006 + shift for i in range(0,24)] + [a.pdf(i / 10.0) + bound + shift for i in range(24,77)] + [b.pdf(i / 10.0) + bound + 0.006 + shift for i in range(77,100)], alpha=0.5, color='grey')

  h3, = plt.plot([0], [0], color='grey', linewidth=20, label='Error Bound')

  # plt.ylim(0.0, 100.0)
  plt.xlabel('State', fontsize=30)
  plt.ylabel('Value', fontsize=30)
  plt.xticks([])
  plt.yticks([])
  # plt.tick_params(axis='x', labelsize=30)
  # plt.tick_params(axis='y', labelsize=30)
  plt.legend(handles=[h1, h2, h3], labels=[r'$\bar{V}^{\pi^{\ast}}$', r'$\bar{V}^{\pi^{\ast}_{I}}$', 'Error Bound'], fontsize=30, loc='best')
  # plt.legend()
  plt.show()
  plt.clf()


def main():
  f()


if __name__ == "__main__":
  main()
