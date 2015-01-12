#ifndef REDUCE_COMMON_H
#define REDUCE_COMMON_H

#include "bvh_common.h"

#define ReduceMaxBlocks 64
#define ReduceMaxThreads 512

static void getReduceBlockThread(uint & blocks, uint & threads, uint n)
{
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

