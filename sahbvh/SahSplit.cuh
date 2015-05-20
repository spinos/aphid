#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"
#include "sah_math.cuh"

namespace sahsplit {

struct DataInterface {
    int2 * nodes;
    Aabb * nodeAabbs;
    KeyValuePair * primitiveIndirections;
    Aabb * primitiveAabbs;
};

struct SplitTask {
template <typename QueueType, int NumBins, int NumThreads>
    __device__ int execute(QueueType * q,
                            DataInterface data,
                            int * smem)
    {
        int & sWorkPerBlock = smem[0];
        int headToSecond, spawn;

        computeBestBin<NumBins, NumThreads>(sWorkPerBlock, data, smem);

        if(threadIdx.x == 0) {             
                /*
                headToSecond = (root.y - root.x + 1) / 2;
                
                if((root.y - root.x + 1) > 256*2) {               
                        spawn = q->enqueue();
                        data.nodes[spawn].x = root.x;
                        data.nodes[spawn].y = headToSecond - 1;

                        spawn = q->enqueue();
                        data.nodes[spawn].x = headToSecond;
                        data.nodes[spawn].y = root.y;
                }*/
                    
                q->setWorkDone();
                q->swapTails();

        }
       // __threadfence();
       // __syncthreads();
        
        return 1;   
    }
   
template<int NumBins, int NumThreads>
    __device__ void computeBestBin(int iRoot,
                                    DataInterface data,
                                    int * smem)
    {
        int2 root = data.nodes[iRoot];
        Aabb rootBox = data.nodeAabbs[iRoot];
        
        int * sSide = &smem[1];
        float * sCost = (float *)&smem[1 + NumBins * NumThreads];
        SplitBin * sBin = (SplitBin *)&smem[1 + NumBins * NumThreads + NumBins];
        SplitBin * sBestBin = (SplitBin *)&smem[1 + NumBins * NumThreads + NumBins
                                                    + NumBins * SIZE_OF_SPLITBIN];
        
/*
 *    layout of sides
 *    0    n     2n    3n
 *    1    n+1   2n+1  3n+1
 *   
 *    n-1  2n-1  3n-1  4n-1
 *
 *    vertical computeSides
 *    horizonal collectBins
 */
        int * sideVertical = &sSide[NumBins * threadIdx.x];
        int * sideHorizontal = &sSide[threadIdx.x];
        
        computeBestBinPerDimension<NumBins, NumThreads, 0>(sBestBin,
                                    sideVertical,
                                    sideHorizontal,
                                    sCost,
                                    data.primitiveIndirections,
                                    data.primitiveAabbs,
                                    sBin,
                                    root,
                                    rootBox);
        
        computeBestBinPerDimension<NumBins, NumThreads, 1>(sBestBin,
                                    sideVertical,
                                    sideHorizontal,
                                    sCost,
                                    data.primitiveIndirections,
                                    data.primitiveAabbs,
                                    sBin,
                                    root,
                                    rootBox);
        
        computeBestBinPerDimension<NumBins, NumThreads, 2>(sBestBin,
                                    sideVertical,
                                    sideHorizontal,
                                    sCost,
                                    data.primitiveIndirections,
                                    data.primitiveAabbs,
                                    sBin,
                                    root,
                                    rootBox);
        
        if(threadIdx.x < 1) {
            int bestI = sBestBin[0].id;
            if(sBestBin[1].plane < sBestBin[0].plane) {
                sBestBin[0] = sBestBin[1];
                bestI = sBestBin[1].id;
            }
            
            if(sBestBin[2].plane < sBestBin[0].plane) {
                sBestBin[0] = sBestBin[2];
                bestI = sBestBin[2].id;
            }
// first is the best
            sBestBin[0].plane = splitPlaneOfBin(&rootBox,
                                                NumBins,
                                                bestI);
        }
    }
    
template<int NumBins, int NumThreads, int Dimension>
    __device__ void computeBestBinPerDimension(SplitBin * sBestBin,
                                    int * sideVertical,
                                    int * sideHorizontal,
                                    float * sCost,
                                    KeyValuePair * primitiveIndirections,
                                    Aabb * primitiveAabbs,
                                    SplitBin * sBin,
                                    int2 root,
                                    Aabb rootBox)
    {
        if(threadIdx.x < NumBins) {
            resetSplitBin(sBin[threadIdx.x]);
        }
        
        __syncthreads();
        
        int nbatch = (root.y - root.x + 1)/NumThreads;
        if(( root.y - root.x + 1) & (NumThreads-1)) nbatch++;

        int i=0;
        for(;i<nbatch;i++) {
            computeSides<NumBins, Dimension>(sideVertical,
                       rootBox,
                       primitiveIndirections,
                       primitiveAabbs,
                       root.x + i * NumThreads,
                       root.y);
        
            __syncthreads();
        
            if(threadIdx.x < NumBins) {
                collectBins<NumBins, NumThreads>(sBin[threadIdx.x],
                    primitiveIndirections,
                    primitiveAabbs,
                    sideHorizontal,
                    root.x + i * NumThreads,
                    root.y);
            }
    
            __syncthreads();
        }
        
        if(threadIdx.x < NumBins) {
            float rootArea = areaOfAabb(&rootBox);
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
        }
        
         __syncthreads();
        
        if(threadIdx.x < 1) {
            int bestI = 0;
            float lowestCost = sCost[0];
            for(i=0; i< NumBins; i++) {
                if(lowestCost > sCost[i]) {
                    lowestCost = sCost[i];
                    bestI = i;
                }
            }
            sBestBin[Dimension] = sBin[bestI];
// store cost here
            sBestBin[Dimension].plane = sCost[bestI];//splitPlaneOfBin(&rootBox,
                                         //       NumBins,
                                         //       bestI);
            sBestBin[Dimension].id = Dimension * NumBins + bestI;
        }
        
        __syncthreads();
    }
};

template <typename QueueType, typename TaskType, typename TaskData, int IdelLimit, int NumBins, int NumThreads>
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
            task.template execute<QueueType, NumBins, NumThreads>(q, data, smem);
        } else {
            q->advanceStopClock();
            i--;
        } 
    }
}

}
