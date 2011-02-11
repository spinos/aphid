#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("hello", ["customer.cpp", "hellomodule.cpp"],
        libraries = ["boost_python"],
        include_dirs = ["/Users/jianzhang/Library/boost_1_44_0"],
        library_dirs = ["/Users/jianzhang/Library/boost_1_44_0/stage/lib"])
    ])
