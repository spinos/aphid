#ifndef REDUCE_COMMON_H
#define REDUCE_COMMON_H

#include "bvh_common.h"

#define ReduceMaxBlocks 128
#ifdef CUDA_V3
#define ReduceMaxThreads 256
#else
#define ReduceMaxThreads 512
#endif

static uint factorBy2(uint x)
{
    uint y = 4;
    uint b = (x + y - 1) / y;
    while(y < b) {
        y = y << 1;
        b = (x + y - 1) / y;
    }
    return y;
}

static void getReduceBlockThread(uint & blocks, uint & threads, uint n)
{
// n must be power of 2
    threads = (n < ReduceMaxThreads*2) ? nextPow2((n + 1)/ 2) : ReduceMaxThreads;	
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	if(blocks > ReduceMaxBlocks) blocks = ReduceMaxBlocks;
}

static unsigned getReduceLastNThreads(unsigned n)
{
	unsigned threads, blocks;
	getReduceBlockThread(blocks, threads, n);
	
	n = blocks;
	while(n > 1) {
		getReduceBlockThread(blocks, threads, n);
		
		n = (n + (threads*2-1)) / (threads*2);
	}
	return threads;
}
#endif        //  #ifndef REDUCE_COMMON_H

