#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"
#include "radixsort_implement.h"

namespace sahsplit {

struct DataInterface {
    int2 * nodes;
    Aabb * nodeAabbs;
    KeyValuePair * primitiveIndirections;
    Aabb * primitiveAabbs;
};

struct SplitTask {
template <typename QueueType>
    __device__ int execute(QueueType * q, 
                            DataInterface data,
                            int * smem)
    {
        int & sWorkPerBlock = smem[0];
        int2 root;
        int headToSecond, spawn;
        
        root = data.nodes[sWorkPerBlock];
        
         
        __syncthreads();
        
        if(threadIdx.x == 0) {             
                
                headToSecond = (root.y - root.x + 1) / 2;
                
                if((root.y - root.x + 1) > 256*2) {               
                        spawn = q->enqueue();
                        data.nodes[spawn].x = root.x;
                        data.nodes[spawn].y = headToSecond - 1;

                        spawn = q->enqueue();
                        data.nodes[spawn].x = headToSecond;
                        data.nodes[spawn].y = root.y;
                }
                    
                q->setWorkDone();
                q->swapTails();

        }
       // __threadfence();
       // __syncthreads();
        
        return 1;
  } 
};

template <typename QueueType, typename TaskType, typename TaskData, int IdelLimit>
__global__ void work_kernel(QueueType * q,
                        TaskType task,
                        TaskData data,
                        int loopLimit, 
                        int workLimit)
{
    extern __shared__ int smem[]; 
    
    int & sWorkPerBlock = smem[0];
    
    int i;

    for(i=0;i<loopLimit;i++) {
        if(q->template isDone<IdelLimit>(workLimit)) break;
        
        if(threadIdx.x == 0) {
            sWorkPerBlock = q->dequeue();
        }     
        __syncthreads();
        
        if(sWorkPerBlock>-1) {
            task.execute(q, data, smem);
        } else {
            q->advanceStopClock();
            i--;
        } 
    }
}

}
