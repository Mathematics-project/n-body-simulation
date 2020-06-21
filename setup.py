from setuptools import setup
from Cython.Build import cythonize

setup(
      name = 'my app',
      ext_modules = cythonize("Math_Project_Cython.pyx"),
      zip_safe = False,
      )

""" Steps to use Cython:
    (1) Run this (setup.py) file to create c file of Math_Project.c
    (2) Use the Anaconda Prompt or other python prompt to create python
        compiled file with line : python setup.py build_ext --inplace
        Math_Project_cython.cp37-win_amd64.pyd file created.
    (3) Run performance_analysis.py file to use the enhanced code
        and make sure you import the Math_Project_Cython """