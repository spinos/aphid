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
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <radixsort_implement.h>
#include <algorithm>
#include <math.h>

void makeRandomFloatVector(float *a, unsigned int numElements)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        a[j] = pow(-1,(float)j)*(float)((rand()<<16) | rand());
    }
}

void makeRandomUintVector(KeyValuePair *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i].key = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
		a[i].value = i;
    }
}

void testSort(int argc, char **argv)
{
    int cmdVal;
    int keybits = 30;

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
        printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits, "unsigned int key-value pairs");

    int deviceID = -1;
    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = 2 * numElements * sizeof(KeyValuePair);
        if (devprop.totalGlobalMem < totalMem)
        {
            printf("Error: not enough memory to sort %d elements.\n", numElements);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(0);
        }
		printf("required devoice memory %d bytes\n", totalMem);
    }

    KeyValuePair *h_data = (KeyValuePair *)malloc(numElements*sizeof(KeyValuePair));
    
    makeRandomUintVector(h_data, numElements, keybits);
	
	for(unsigned int i=0; i<numElements-1; ++i)
    {
		// printf("%4d : %4d\n", h_data[i].key, h_data[i].value);
	}

    KeyValuePair *d_data;
    cudaMalloc((void **)&d_data, numElements*sizeof(KeyValuePair));
	
	cudaMemcpy(d_data, h_data, numElements * sizeof(KeyValuePair), cudaMemcpyHostToDevice);
    
	KeyValuePair *d_data1;
    cudaMalloc((void **)&d_data1, numElements*sizeof(KeyValuePair));
	
    RadixSort(d_data, d_data1, numElements, 32);
    
    cudaMemcpy(h_data, d_data, numElements * sizeof(KeyValuePair), cudaMemcpyDeviceToHost);
    
	char passed = 1;
    for(unsigned int i=0; i<numElements-1; ++i)
    {
		// printf("%4d : %4d\n", h_data[i].key, h_data[i].value);
        if( (h_data[i].key)>(h_data[i+1].key) )
        {
            printf("Unordered key[%d]: %d > key[%d]: %d\n", i, h_data[i].key, i+1, h_data[i+1].key);
            passed = 0;
			break;
	    }
    }
	
	if(passed) printf("passed!\n");

    cutilSafeCall( cudaFree(d_data) );
    cutilSafeCall( cudaFree(d_data1) );
    free(h_data);
}

int main(int argc, char **argv)
{
    cutilDeviceInit(argc, argv);
  
    testSort(argc, argv);

	cutilExit(argc, argv);
	return EXIT_SUCCESS;
}


