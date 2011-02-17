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

// -----------------------------------------------------------------------
// Fast CUDA Radix Sort 
//
// The parallel radix sort algorithm implemented by this code is described
// in the following paper.
//
// Satish, N., Harris, M., and Garland, M. "Designing Efficient Sorting 
// Algorithms for Manycore GPUs". In Proceedings of IEEE International
// Parallel & Distributed Processing Symposium 2009 (IPDPS 2009).
//
// -----------------------------------------------------------------------


// This file is a test rig for the RadixSort class.  It can run radix sort on 
// random arrays using various command line options, listed below.
//
// COMMAND LINE OPTIONS
// -n=<N>        : number of elements to sort
// -keysonly     : sort only an array of keys (the default is to sort key-value pairs)
// -float        : use 32-bit float keys
// -keybits=<B>  : Use only the B least-significant bits of the keys for the sort
//               : B must be a multiple of 4.  This option does not apply to float keys
// -quiet        : Output only the number of elements and the time to sort
// -help         : Output a help message

#include <stdio.h>
#include <cutil_inline.h>
#include <algorithm>
#include <math.h>

#include "radixsort.h"

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
bool verifySortUint(unsigned int *keysSorted, 
		    unsigned int *valuesSorted, 
		    unsigned int *keysUnsorted, 
		    unsigned int len)
{
    bool passed = true;
    for(unsigned int i=0; i<len-1; ++i)
    {
        if( (keysSorted[i])>(keysSorted[i+1]) )
	{
	    printf("Unordered key[%d]: %d > key[%d]: %d\n", i, keysSorted[i], i+1, keysSorted[i+1]);
	    passed = false;
	    break;
	}
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u): %u != %u\n", 
		       i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                passed = false;
                break;
            }
        }
    }

    return passed;
}

bool verifySortFloat(float        *keysSorted, 
		     unsigned int *valuesSorted, 
		     float        *keysUnsorted, 
		     unsigned int len)
{
    bool passed = true;
    for(unsigned int i=0; i<len-1; ++i)
    {
        if( (keysSorted[i])>(keysSorted[i+1]) )
	{
	    printf("Unordered key[%d]: %f > key[%d]: %f\n", i, keysSorted[i], i+1, keysSorted[i+1]);
	    passed = false;
	    break;
	}
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u): %f != %f\n", 
		       i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                passed = false;
                break;
            }
        }
    }

    return passed;
}

void makeRandomFloatVector(float *a, unsigned int numElements)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        a[j] = pow(-1,(float)j)*(float)((rand()<<16) | rand());
    }
}



void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}


template <typename T, bool floatKeys>
void testSort(int argc, char **argv)
{
    int cmdVal;
    int keybits = 32;

#ifdef __DEVICE_EMULATION__
    unsigned int numElements = 3500;
#else
    unsigned int numElements = 1048576;
#endif

    bool keysOnly = (cutCheckCmdLineFlag(argc, (const char**)argv, "keysonly") == CUTTrue);

    bool quiet = (cutCheckCmdLineFlag(argc, (const char**)argv, "quiet") == CUTTrue);

    if( cutGetCmdLineArgumenti( argc, (const char**)argv, "n", &cmdVal) )
    { 
        numElements = cmdVal;
    }

    if( cutGetCmdLineArgumenti( argc, (const char**)argv, "keybits", &cmdVal) )
    {
        keybits = cmdVal;
    }
#ifdef __DEVICE_EMULATION__
    unsigned int numIterations = 1;
#else
    unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;
#endif

    if ( cutGetCmdLineArgumenti(argc, (const char**) argv, "iterations", &cmdVal) )
    {
        numIterations = cmdVal;
    }

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "help") )
    {
        printf("Command line:\nradixsort_block [-option]\n");
        printf("Valid options:\n");
        printf("-n=<N>        : number of elements to sort\n");
        printf("-keysonly     : sort only an array of keys (the default is to sort key-value pairs)\n");
        printf("-float        : use 32-bit float keys (default is 32-bit unsigned int)\n");
        printf("-keybits=<B>  : Use only the B least-significant bits of the keys for the sort\n");
        printf("              : B must be a multiple of 4.  This option does not apply to float keys\n");
        printf("              : B must be a multiple of 4.  This option does not apply to float keys\n");
        printf("-quiet        : Output only the number of elements and the time to sort\n");
        printf("-help         : Output a help message\n");
        exit(0);
    }

    if (!quiet)
        printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits, floatKeys ? "float" : "unsigned int", keysOnly ? "(only)" : "and values");

    int deviceID = -1;
    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = (keysOnly ? 2 : 4) * numElements * sizeof(T);
        if (devprop.totalGlobalMem < totalMem)
        {
            printf("Error: not enough memory to sort %d elements.\n", numElements);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(0);
        }
    }

    T *h_keys       = (T*)malloc(numElements*sizeof(T));
    T *h_keysSorted = (T*)malloc(numElements*sizeof(T));
    unsigned int *h_values     = 0;
    if (!keysOnly)
        h_values = (unsigned int *)malloc(numElements*sizeof(unsigned int));

    // Fill up with some random data
    if (floatKeys)
    {
        makeRandomFloatVector((float*)h_keys, numElements);
    }
    else
    {
        makeRandomUintVector((unsigned int*)h_keys, numElements, keybits);
    }
    
    if (!keysOnly)
    {
        for(unsigned int i=0; i < numElements; ++i)   
        { 
            h_values[i] = i; 
        }
    }

    // Copy data onto the GPU
    T *d_keys;
    unsigned int *d_values;
    cudaMalloc((void **)&d_keys, numElements*sizeof(T));
    if (!keysOnly)
        cudaMalloc((void **)&d_values, numElements*sizeof(unsigned int));
    else
        d_values = 0;

    // Creat the RadixSort object
    RadixSort radixsort(numElements, keysOnly);

    cudaMemcpy(d_keys, h_keys, numElements * sizeof(T), cudaMemcpyHostToDevice);
    if (!keysOnly)
        cudaMemcpy(d_values, h_values, numElements * sizeof(unsigned int), cudaMemcpyHostToDevice);

#ifndef __DEVICE_EMULATION__
    // run once to warm up (so we don't time setup)
    if (floatKeys)
        radixsort.sort((float*)d_keys, d_values, numElements, keybits, true);
    else
        radixsort.sort((unsigned int*)d_keys, d_values, numElements, keybits);
#endif

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );

    float totalTime = 0;
    for(unsigned int i = 0; i < numIterations; i++) 
    {
        // reset data before sort
        cudaMemcpy(d_keys, h_keys, numElements * sizeof(T), cudaMemcpyHostToDevice);
	    if (!keysOnly)
	        cudaMemcpy(d_values, h_values, numElements * sizeof(unsigned int), cudaMemcpyHostToDevice);

	    cutilSafeCall( cudaEventRecord(start_event, 0) );

        if (floatKeys)
            radixsort.sort((float*)d_keys, d_values, numElements, keybits, true);
        else
            radixsort.sort((unsigned int*)d_keys, d_values, numElements, keybits);

	    cutilSafeCall( cudaEventRecord(stop_event, 0) );
	    cutilSafeCall( cudaEventSynchronize(stop_event) );

	    float time = 0;
	    cutilSafeCall( cudaEventElapsedTime(&time, start_event, stop_event));
	    totalTime += time;
    }

    CUT_CHECK_ERROR("after radixsort");

    // Get results back to host for correctness checking
    cudaMemcpy(h_keysSorted, d_keys, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    if (!keysOnly)
        cudaMemcpy(h_values, d_values, numElements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    else
        h_values = 0;

    CUT_CHECK_ERROR("copying results to host memory");

    // Check results
    bool passed = true;
    if (floatKeys)
        passed = verifySortFloat((float*)h_keysSorted, h_values, (float*)h_keys, numElements);
    else
        passed = verifySortUint((unsigned int*)h_keysSorted, h_values, (unsigned int*)h_keys, numElements);

    if( !passed && !quiet )  
        printf("Sorting : FAIL\n");

    if(passed)
    {
        if (!quiet)
        {
            printf("Sorting : PASS\n");        
            printf(":  %-15s %-15s\n", "elements",  "GPUms");
        }
        printf(":: %-15d %-15.8f\n", numElements, totalTime*1.0/numIterations);
    }

    cutilSafeCall( cudaEventDestroy(start_event) );
    cutilSafeCall( cudaEventDestroy(stop_event) );
    cutilSafeCall( cudaFree(d_keys) );
    cutilSafeCall( cudaFree(d_values) );
    free(h_keys);
    free(h_values);
}

int main(int argc, char **argv)
{
    cutilDeviceInit(argc, argv);
  
    if( cutCheckCmdLineFlag( argc, (const char**)argv, "float") )
        testSort<float, true>(argc, argv);
    else
        testSort<unsigned int, false>(argc, argv);

	cutilExit(argc, argv);
	return EXIT_SUCCESS;
}


