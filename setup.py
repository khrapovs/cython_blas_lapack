from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(name='cython_blas_test',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("cython_functions", ["cython_functions.pyx"],
                             include_dirs=[numpy.get_include()],), ]
      )
