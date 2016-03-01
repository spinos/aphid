#ifndef IMAGEBASE_CUH
#define IMAGEBASE_CUH

#include "cu/AllBase.h"

inline __device__ uint encodeARGB(int r, int g, int b, int a)
{ return ((a<<24) | (r<<16) | (g<<8) | b); }

inline __device__ uint encodeRGB(int r, int g, int b)
{ return ((255<<24) | (r<<16) | (g<<8) | b); }

#endif        //  #ifndef IMAGEBASE_CUH

