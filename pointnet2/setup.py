from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

import sys
sys.path.append("..")
import builtins
builtins.__POINTNET2_SETUP__ = True
import pointnet2

_ext_src_root = "./_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext-src', 'include')

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='_ext',
            sources=_ext_sources,
            extra_compile_args={
                 "cxx": ["-O2", headers],
                 "nvcc": ["-O2", headers]
                # "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                # "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
