#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

template <typename QueueType, typename TaskType, typename TaskData, int IdelLimit>
__global__ void quickSort_test_kernel(QueueType * q,
                        TaskType task,
                        TaskData data,
                        SimpleQueueInterface * qi,
                        uint * workBlocks,
                        uint * loopbuf,
                        int4 * headtailperloop,
                        int loopLimit,
                        int workLimit)
{
    extern __shared__ int smem[]; 
    
    int & sWorkPerBlock = smem[0];
    
    int i;
    int loaded = 0;
    
    for(i=0;i<loopLimit;i++) {
        if(q->template isDone<IdelLimit>(workLimit)) break;
        
        if(threadIdx.x == 0) {
            sWorkPerBlock = q->dequeue();
        }     
        __syncthreads();
        
        if(sWorkPerBlock>-1) {
            task.execute(q, data, smem);
// for debug purpose only
            atomicMax(&qi->workBlock, blockIdx.x);
            workBlocks[sWorkPerBlock] = blockIdx.x;
            loaded++;
        } else {
            q->advanceStopClock();
            i--;
        }

// for debug purpose only
        if(threadIdx.x <1) {
            loopbuf[blockIdx.x] = loaded; 
            headtailperloop[i].x= q->head();
            headtailperloop[i].y= q->intail();
            headtailperloop[i].z= q->outtail();
            headtailperloop[i].w= q->workDoneCount();
            qi->qhead= q->head();
            qi->qintail= q->intail();
            qi->qouttail= q->outtail();
            qi->workDone= q->workDoneCount();
            qi->lastBlock = blockIdx.x;
        } 
    }
}

#endif        //  #ifndef QUICKSORT_CUH

