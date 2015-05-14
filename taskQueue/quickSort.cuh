#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

__device__ void quickSort_redistribute(uint * data,
                            int2 range,
                            int & headToSecond)
{
    int low = range.x;
    int high = range.y;
    headToSecond = (low + high)/2;
    
    if(low > high) return;
    
    uint intermediate;
    uint separator = data[(low + high)/2];
    for(;;) {
        while(data[low]<separator) low++;
        while(data[high]>separator) high--;
            
        if(low <= high) {
            intermediate = data[low];
            data[low++] = data[high];
            data[high--] = intermediate;
        }
        
        if(low > high) break;
    }
    
    headToSecond = low;
}

__global__ void quickSort_checkQ_kernel(uint * maxN,
                        simpleQueue::SimpleQueue * q,
                        SimpleQueueInterface * qi,
                        uint * idata,
                        int2 * nodes,
                        uint * workBlocks,
                        uint * loopbuf,
                        int4 * headtailperloop)
{
    __shared__ int sWorkPerBlock[512];
    // __shared__ simpleQueue::TimeLimiter<1000> timelimiter;

    int i=0, j;
    int loaded = 0;
    int2 root;
    int headToSecond, spawn, offset;
    
    for(i=0;i<9;i++) 
    {
        if(i<1) {
            q->init(qi);
            continue;
        }
        
        if(q->isEmpty()) {
            continue;
        }
        
       // if(threadIdx.x < 1)
         //   for(j=0;j<(blockDim.x>>5);j++) sWorkPerBlock[j] = 1;

        __syncthreads();

        if((threadIdx.x&31) == 0) {
            sWorkPerBlock[threadIdx.x>>5] = q->dequeue();
            
            if(sWorkPerBlock[threadIdx.x>>5] > -1) { loaded++;
                root = nodes[sWorkPerBlock[threadIdx.x>>5]];
                
                workBlocks[sWorkPerBlock[threadIdx.x>>5]] = blockIdx.x;//q->workDoneCount();//offset;//q->tail() - q->head();//
                
                quickSort_redistribute(idata,
                            root,
                            headToSecond);
                    
                    if(root.x +1 < root.y) 
                    {
                      if(root.x < headToSecond - 1) {
                        spawn = q->enqueue();
                        nodes[spawn].x = root.x;
                        nodes[spawn].y = headToSecond - 1;
                      }
                      
                      
                      
                      if(headToSecond < root.y) {
                        spawn = q->enqueue();
                        nodes[spawn].x = headToSecond;
                        nodes[spawn].y = root.y;
                      }
                      
                      
                    }
                    
                    q->setWorkDone();
                    qi->workBlock = spawn;
        
                    __threadfence_block();
                    
                    q->swapTails();
            }           
                     
        }
        
        //if(q->workDoneCount() == q->tail() && q->tail() > 1) {
          // qi->qtail = q->tail();
          // timelimiter.start();
        //}
        __syncthreads();
        
        //if(timelimiter.stop()) break;
        
        if(threadIdx.x < 1) {
        loopbuf[blockIdx.x] = loaded; 
        
        }
        
        //
          //  
                  
        if(threadIdx.x <1) {
            headtailperloop[i].x = q->head();
            headtailperloop[i].y= q->intail();
            headtailperloop[i].z= q->outtail();
            headtailperloop[i].w= q->workDoneCount();
        }
    }
    
    if(blockIdx.x < 1 && threadIdx.x <1) {
             *maxN = i;//q->workDoneCount();
        }
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

