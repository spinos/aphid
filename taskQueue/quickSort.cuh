#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

template <typename QueueType, typename TaskType, int LoopLimit, int WorkLimit, int IdelLimit>
__global__ void quickSort_checkQ_kernel(QueueType * q,
                        TaskType task,
                        SimpleQueueInterface * qi,
                        uint * idata,
                        int2 * nodes,
                        uint * workBlocks,
                        uint * loopbuf,
                        int4 * headtailperloop)
{
    extern __shared__ int smem[]; 
    
    int & sWorkPerBlock = smem[0];
    
    int i;
    int loaded = 0;
    
    for(i=0;i<LoopLimit;i++) {
        if(q->template isDone<WorkLimit, IdelLimit>()) break;
        
        if(task.execute(q, smem, idata, nodes)) {
// for debug purpose only
            qi->workBlock = blockIdx.x;
            workBlocks[sWorkPerBlock] = blockIdx.x;
            loaded++;
        }

        __syncthreads();
                  
        if(threadIdx.x < 1) q->swapTails();

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

