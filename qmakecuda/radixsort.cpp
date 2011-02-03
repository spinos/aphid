/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
 
 #include <map>
#include <cuda_runtime.h>
#include <assert.h>
#include <algorithm>
#include "radixsort.h"

// Used for creating a mapping of kernel functions to the number of 
// CTAs to launch for each
typedef void* KernelPointer;
std::map<KernelPointer, int> numCTAsTable;

extern "C" 
int getNumCTAs(KernelPointer kernel)
{
    return numCTAsTable[kernel];
}
extern "C" 
void setNumCTAs(KernelPointer kernel, int numCTAs)
{
    numCTAsTable[kernel] = numCTAs;
}

// computes next highest multiple of f from x
inline size_t multiple(size_t x, size_t f)
{
    return ((x + (f-1)) / f);
}


// MS Excel-style CEIL() function
// Rounds x up to nearest multiple of f
inline size_t ceiling(size_t x, size_t f)
{
    return multiple(x, f) * f;
}

extern "C"
void computeNumCTAs(KernelPointer kernel, int smemDynamicBytes, bool bManualCoalesce)
{
    cudaDeviceProp devprop;
    int deviceID = -1;
    cudaError_t err = cudaGetDevice(&deviceID);
    assert(err == cudaSuccess);

    cudaGetDeviceProperties(&devprop, deviceID);

    // Determine the maximum number of CTAs that can be run simultaneously for each kernel
    // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet
    const unsigned int regAllocationUnit = (devprop.major < 2 && devprop.minor < 2) ? 256 : 512; // in registers
    const unsigned int warpAllocationMultiple = 2;
    const unsigned int smemAllocationUnit = 512;                                                 // in bytes
    const unsigned int maxThreadsPerSM = bManualCoalesce ? 768 : 1024; // sm_12 GPUs increase threads/SM to 1024
    const unsigned int maxBlocksPerSM = 8;

    cudaFuncAttributes attr;
    err = cudaFuncGetAttributes(&attr, (const char*)kernel);
    assert(err == cudaSuccess);


    // Number of warps (round up to nearest whole multiple of warp size)
    size_t numWarps = multiple(RadixSort::CTA_SIZE, devprop.warpSize);
    // Round up to warp allocation multiple
    numWarps = ceiling(numWarps, warpAllocationMultiple);

    // Number of regs is regs per thread times number of warps times warp size
    size_t regsPerCTA = attr.numRegs * devprop.warpSize * numWarps;
    // Round up to multiple of register allocation unit size
    regsPerCTA = ceiling(regsPerCTA, regAllocationUnit);

    size_t smemBytes = attr.sharedSizeBytes + smemDynamicBytes;
    size_t smemPerCTA = ceiling(smemBytes, smemAllocationUnit);

    size_t ctaLimitRegs    = regsPerCTA > 0 ? devprop.regsPerBlock / regsPerCTA : maxBlocksPerSM;
    size_t ctaLimitSMem    = smemPerCTA > 0 ? devprop.sharedMemPerBlock      / smemPerCTA : maxBlocksPerSM;
    size_t ctaLimitThreads =                  maxThreadsPerSM                / RadixSort::CTA_SIZE;

    unsigned int numSMs = devprop.multiProcessorCount;
    int maxCTAs = numSMs * std::min<size_t>(ctaLimitRegs, std::min<size_t>(ctaLimitSMem, std::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
    setNumCTAs(kernel, maxCTAs);
}

