/*
 *  odd-even sort
 0:  67   67   67   25   25   25   25   25   25   18
 1:  138  138  25   67   52   52   52   52   18   25
 2:  52   25   138  52   67   67   67   18   52   52
 3:  25   52   52   138  133  133  18   67   67   67
 4:  133  133  133  133  138  18   133  133  133  122
 5:  207  207  151  151  18   138  138  138  122  133
 6:  158  151  207  18   151  151  151  122  138  138
 7:  151  158  18   207  152  152  122  151  144  144
 8:  159  18   158  152  207  122  152  144  151  151
 9:  18   159  152  158  122  207  144  152  152  152
 10:  242 152  159  122  158  144  207  158  158  158
 11:  152 242  122  159  144  158  158  207  159  159
 12:  122 122  242  144  159  159  159  159  207  207
 13:  220 144  144  242  220  220  220  220  220  220
 14:  144 220  220  220  242  242  242  242  242  242
odd/even  0    1    0    1    0    1    0    1    0   1 
sorted    0    0    0    0    0    0    0    0    0   1
 *
 */

#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

__device__ void quickSort_oddEven(uint * data,
                            int low, int high, 
                            int & sorted, int & isOdd)
{
    if(threadIdx.x ==0) {
        sorted = 0;
        isOdd = 0;
    }
    
    __syncthreads();
    
    uint intermediate;
    int left, right;
    while(sorted<1) {
        
        if(threadIdx.x ==0) {
            sorted = 1;
        }
        
        __syncthreads();
        
        left = low + threadIdx.x * 2 + isOdd;
        right = left + 1;
        if(right <= high) {
            if(data[right] < data[left]) {
                intermediate = data[right];
                data[right] = data[left];
                data[left] = intermediate;
                sorted = 0;
            }
        }
        
        __syncthreads();
        
        if(threadIdx.x ==0) {
            isOdd = (isOdd + 1)&1;
        }
        
        __syncthreads();
    }
}

__device__ void quickSort_swapHead(uint * data,
                            int low, int high)
{
    int headToSecond = (low + high)/2+1;
    if(headToSecond > high) return;
    
    uint intermediate;
    int i, j;
    
    for(j=0;j<headToSecond-2;j++){
// highest in left
        i=low;
        for(;i<headToSecond-1;i++) {
            if(data[headToSecond-1] < data[i]) {
                intermediate = data[i];
                data[i] = data[headToSecond-1];
                data[headToSecond-1] = intermediate;
            }
        }

// lowest in right
        i=headToSecond+1;
        for(;i<=high;i++) {
            if(data[headToSecond] > data[i]) {
                intermediate = data[i];
                data[i] = data[headToSecond];
                data[headToSecond] = intermediate;
            }
        }

// swap at split 
        if(data[headToSecond-1] > data[headToSecond]) {
            intermediate = data[headToSecond-1];
            data[headToSecond-1] = data[headToSecond];
            data[headToSecond] = intermediate;
        }
        else break;
    }
}

__device__ void quickSort_redistribute(uint * data,
                            int low, int high)
{
    int headToSecond = (low + high)/2+1;
    
    if(headToSecond > high) return;
    
    uint intermediate;
    
    if(headToSecond == high) {
        if(threadIdx.x > 0) return;
        if(data[high] < data[low]) {
                intermediate = data[low];
                data[low] = data[high];
                data[high] = intermediate;
        }
        return;
    }
    
    int i = threadIdx.x;
    
    while((low + i)<headToSecond) {
        if((headToSecond + i) <= high) {
            if(data[headToSecond + i] < data[low+i]) {
                intermediate = data[low+i];
                data[low+i] = data[headToSecond + i];
                data[headToSecond + i] = intermediate;
            }
        }
        
        i += blockDim.x;
    }
}

template <int LoopLimit, int WorkLimit>
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
    __shared__ int sSorted;
    __shared__ int sOdd;
    // __shared__ simpleQueue::TimeLimiter<1000> timelimiter;

    int i=0, j;
    int loaded = 0;
    int2 root;
    int headToSecond, spawn, offset;
    
    if(threadIdx.x <1) {
        if(q->isEmpty()) q->init<WorkLimit>(qi);
    }
    
    __syncthreads();
    
    for(i=0;i<LoopLimit;i++) 
    {
        if(q->isDone<WorkLimit>()) break;
       
        if(threadIdx.x <1) {
            headtailperloop[i].x = 0;
            headtailperloop[i].y= 0;
            headtailperloop[i].z = 0;
            headtailperloop[i].w= 0;
        }
        
        if((threadIdx.x) == 0) {
            sWorkPerBlock[0] = q->dequeue();
        }
        
        __syncthreads();
        
        if(threadIdx.x ==0) {
            if(sWorkPerBlock[0] > -1) {
                root = nodes[sWorkPerBlock[0]];
                headToSecond = (root.x + root.y)/2+1;
                if((root.y - root.x + 1) > blockDim.x*2) 
                        quickSort_swapHead(idata,
                                root.x,
                                root.y);
            }   
        }
        
        __syncthreads();
        
        if(sWorkPerBlock[0] > -1) {
              root = nodes[sWorkPerBlock[0]];
              if((root.y - root.x + 1) <= blockDim.x*2) 
              quickSort_oddEven(idata,
                               root.x,
                               root.y,
                                sSorted,
                                sOdd);
        }
         
        __syncthreads();
        
        if((threadIdx.x) == 0) {
            if(sWorkPerBlock[0] > -1) { loaded++;
                if((root.y - root.x + 1) > blockDim.x*2) {
                //if(root.x < headToSecond - 1) {
                        spawn = q->enqueue();
                        nodes[spawn].x = root.x;
                        nodes[spawn].y = headToSecond - 1;
                  //    }

                    //  if(headToSecond < root.y) {
                        spawn = q->enqueue();
                        nodes[spawn].x = headToSecond;
                        nodes[spawn].y = root.y;
                      //}
                }    
                    q->setWorkDone();
                    qi->workBlock = blockIdx.x;
        
                    //__threadfence_block();

                   if(threadIdx.x <1) {
           // && sWorkPerBlock[0] > -1) {
                        //q->extendTail();
                    }
                    workBlocks[sWorkPerBlock[0]] = blockIdx.x;//q->workDoneCount();//offset;//q->tail() - q->head();//
                      
                    //atomicMin(&headtailperloop[i].z, blockIdx.x);
                    //atomicMax(&headtailperloop[i].w, blockIdx.x);
            }           
                     
        }
        
        __syncthreads();
        
        if(threadIdx.x < 1) {
            loopbuf[blockIdx.x] = loaded; 
        
        }
                  
        if(threadIdx.x < 1) q->swapTails();
        
        if(threadIdx.x <1) {
            headtailperloop[i].x= q->head();
            headtailperloop[i].y= q->intail();
            headtailperloop[i].z= q->outtail();
            headtailperloop[i].w= q->workDoneCount();
        }
        
    }
    
    if(threadIdx.x <1)
             *maxN = i;//q->workDoneCount();
      //  }
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

