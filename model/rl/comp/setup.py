import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

source_pattern = 'model/rl/comp/%s.pyx'

asevent_ext = [
    Extension('svf', sources=[source_pattern % "svf"],
              libraries=['m', 'stdc++', 'z'],
              extra_compile_args = ['-O3', '-funroll-loops',
                                    '-std=c++11', '-fopenmp',
                                    '-D__STDC_CONSTANT_MACROS',
                                    '-D__STDC_LIMIT_MACROS', '-w',
                                    '-Wl,-static',],
              extra_link_args=['-lm', '-std=c++11',
                               '-lz', '-fopenmp',],
              language="c++",
              )
    ]

setup(
    name = 'svf',
    ext_modules = cythonize(asevent_ext),
    cmdclass = {'build_ext': build_ext},
)
