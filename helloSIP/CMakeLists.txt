# basic CMake project for helloSIP
# 

cmake_minimum_required (VERSION 2.8)
project (wordLibrary)

include_directories (".")
    
add_library(wordLibrary STATIC word.cpp)

