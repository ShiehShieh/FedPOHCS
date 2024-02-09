from libcpp cimport bool as cbool
from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort, lower_bound
from libc.math cimport sqrt as csqrt, pow as cpow, fmax as cfmax, fabs as cfabs
from libc.string cimport strcmp, strlen
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fprintf, fopen, fclose


cdef extern from '<numeric>' namespace 'std' nogil:
    T accumulate[InputIt, T]( InputIt first, InputIt last, T init );


cdef struct objs:
  cmap[vector[double], double] isd
  cmap[vector[double], cmap[vector[double], double]] rf
  cmap[vector[double], double] vf
  vector[vector[double]] obss
  vector[vector[double]] acts
  int n_states
  int n_actions
  # vector[vector[vector[double]]] tparray
  # vector[vector[unordered_map[int, double]]] tparray
  vector[unordered_map[int, unordered_map[int, double]]] tparray
  unordered_map[int, vector[int]] predecessor
  unordered_map[int, vector[int]] taken
  unordered_map[int, unordered_map[int, vector[int]]] sas
  unordered_map[int, unordered_map[int, vector[int]]] ssa
  vector[vector[double]] oaprob
  # vector[unordered_map[int, double]] oaprob
  double gamma


cdef struct svfobjs:
  cmap[vector[double], double] d
  cmap[vector[double], double] vf
  cmap[vector[double], cmap[vector[double], double]] qf
  cmap[vector[double], cmap[vector[double], double]] af
