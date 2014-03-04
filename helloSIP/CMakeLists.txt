# basic CMake project for helloSIP
# 

cmake_minimum_required (VERSION 2.8)
project (word)

include_directories (".")
    
add_library(word SHARED word.cpp)

