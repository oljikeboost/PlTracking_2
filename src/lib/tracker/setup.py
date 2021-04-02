from setuptools import setup
from Cython.Build import cythonize
import os
# os.system('rm -rf build & rm cython_util.cpython-37m-x86_64-linux-gnu.so & rm cython_util.c')

setup(
    ext_modules = cythonize("crop_util.pyx", annotate=True)
)