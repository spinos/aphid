#ifndef IMAGEBASE_CUH
#define IMAGEBASE_CUH

#include "cu/AllBase.h"

inline __device__ uint encodeARGB(int r, int g, int b, int a)
{ return ((a<<24) | (r<<16) | (g<<8) | b); }

inline __device__ uint encodeRGB(int r, int g, int b)
{ return ((255<<24) | (r<<16) | (g<<8) | b); }

inline __device__ uint getPixelCoordx()
{ return blockIdx.x * blockDim.x + threadIdx.x; }

inline __device__ uint getPixelCoordy()
{ return blockIdx.y * blockDim.y + threadIdx.y; }

inline __device__ uint getTileIdx()
{ return blockIdx.y * gridDim.x + blockIdx.x; }

inline __device__ uint getTiledPixelIdx()
{ return getTileIdx() * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; }

inline __device__ uint getImagePixelIdx(uint x, uint y)
{ return y * blockDim.x * gridDim.x
        + x; }

#endif        //  #ifndef IMAGEBASE_CUH

