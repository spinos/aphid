#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

__global__ void quickSort_checkQ_kernel(uint * maxN,
                        simpleQueue::SimpleQueue * q,
                        SimpleQueueInterface * qi,
                        int2 * nodes)
{
    __shared__ int sWorkPerBlock[128];
    int i=0;
    for(;;) {
        if(blockIdx.x < 1 && threadIdx.x < 1) {
            if(i<1) {
                q->init(qi);
                q->enqueue();
                i++;
            }
        }
        
        if(threadIdx.x < 1) {
            sWorkPerBlock[0] = 1;
            if(q->maxNumWorks() < 1)
                sWorkPerBlock[0] = 0;
        }
        __syncthreads();
        
        if(sWorkPerBlock[0] < 1) continue;

        if(threadIdx.x < 1) {
            sWorkPerBlock[0] = 1;
            if(q->countNewTask() > 0) {
                int workNode = q->dequeue();
                if(workNode >= 0) {
                    qi->workBlock = blockIdx.x;
                    q->setWorkDone();
                }
            }
            else if(q->isAllWorkDone()) sWorkPerBlock[0] = 0;
        }
        __syncthreads();
        
        if(sWorkPerBlock[0] < 1) break;
    }
    
    if(blockIdx.x < 1 && threadIdx.x < 1)
        maxN[0] = q->workDoneCount();
        
}
/*
__global__ void quickSort_kernel(int * obin,
                    int * idata,
                    int h,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	int d = idata[ind];
	
    atomicAdd(&obin[d/h], 1);
}
*/

#endif        //  #ifndef QUICKSORT_CUH

