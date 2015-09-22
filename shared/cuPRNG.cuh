#ifndef CUPRNG_CUH
#define CUPRNG_CUH

/*
 *  reference: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
 *             https://devtalk.nvidia.com/default/topic/456222/cuda-programming-and-performance/random-numbers-inside-the-kernel/
 *             http://www0.cs.ucl.ac.uk/staff/W.Langdon/ftp/papers/langdon_2009_CIGPU.pdf
 */
 
#include <cuda_runtime_api.h>

__device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)  
{ return z=(A*z+C); }

__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)  
{  
  unsigned b=(((z << S1) ^ z) >> S2);  
  return z = (((z & M) << S3) ^ b);
}

__constant__ unsigned int shift1[4] = {6, 2, 13, 3};

__constant__ unsigned int shift2[4] = {13, 27, 21, 12};

__constant__ unsigned int shift3[4] = {18, 2, 7, 13};

__constant__ unsigned int offset[4] = {4294967294, 4294967288, 4294967280, 4294967168};

__device__ unsigned MultiplacationModulus(unsigned & seed)
{
    return seed = (seed * 16807UL) & 2147483647;
}

__device__ float HybridTaus()  
{  
    unsigned z1 = blockIdx.x*blockDim.x + threadIdx.x + 15625;
    unsigned z2 = MultiplacationModulus(z1);
    unsigned z3 = MultiplacationModulus(z2);
    unsigned z4 = MultiplacationModulus(z3);
    return 2.3283064365387e-10f * (              // Periods  
    TausStep(z1, shift1[threadIdx.x & 3], shift2[threadIdx.x & 3], shift3[threadIdx.x & 3], offset[threadIdx.x & 3]) ^
    TausStep(z2, shift1[(threadIdx.x+1) & 3], shift2[(threadIdx.x+1) & 3], shift3[(threadIdx.x+1) & 3], offset[(threadIdx.x+1) & 3]) ^ 
    TausStep(z3, shift1[(threadIdx.x+2) & 3], shift2[(threadIdx.x+2) & 3], shift3[(threadIdx.x+2) & 3], offset[(threadIdx.x+2) & 3]) ^
    TausStep(z4, shift1[(threadIdx.x+3) & 3], shift2[(threadIdx.x+3) & 3], shift3[(threadIdx.x+3) & 3], offset[(threadIdx.x+3) & 3])
   ); 
}

#endif        //  #ifndef CUPRNG_CUH

