#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"
#include "sah_common.h"
#include "bvh_math.cuh"
#include "Aabb.cuh"
#include "onebitsort.cuh"

namespace sahsplit {

struct DataInterface {
    int2 * nodes;
    Aabb * nodeAabbs;
    int * nodeParents;
    int * nodeLevels;
    KeyValuePair * primitiveIndirections;
    Aabb * primitiveAabbs;
    KeyValuePair * intermediateIndirections;
};

struct SplitTask {
    __device__ int shouldSplit(DataInterface data, int iRoot)
    {
        int2 root = data.nodes[iRoot];
        if(root.x>>31) return 0;
        
        return (root.y - root.x) > 7;
    }
    
    __device__ int validateSplit(DataInterface data, int * smem)
    {
        int2 root = data.nodes[smem[0]];
        float * sBestCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
        return (sBestCost[0] < (root.y - root.x-.5f));
    }
    
    template <typename QueueType, int NumBins, int NumThreads>
    __device__ int execute(QueueType * q,
                            DataInterface data,
                            int * smem)
    {
        int sWorkPerBlock = smem[0];
        int doSpawn = 0;
        if(shouldSplit(data, sWorkPerBlock)) {
            computeBestBin<NumBins, NumThreads>(sWorkPerBlock, data, smem);
        
            doSpawn = validateSplit(data, smem);
            if(doSpawn)
                rearrange<NumBins, NumThreads>(data, smem);
        }

        if(threadIdx.x == 0) {  
            if(doSpawn) {
                spawn(q, data, smem);
            }
                
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
/*
 *    layout of memory in int
 *    n  as num bins
 *    t  as num threads
 *    16 as size of bin
 *
 *    0                                      workId
 *    1             -> 1+3*16-1              best bin per dimension
 *    1+3*16        -> 1+3*16+3-1            cost of best bin per dimension
 *
 */             
        computeBestBinPerDimension<NumBins, NumThreads, 0>(data,
                                    smem,
                                    root,
                                    rootBox);
        
        computeBestBinPerDimension<NumBins, NumThreads, 1>(data,
                                    smem,
                                    root,
                                    rootBox);
        
        computeBestBinPerDimension<NumBins, NumThreads, 2>(data,
                                    smem,
                                    root,
                                    rootBox);
        
        SplitBin * sBestBin = (SplitBin *)&smem[1];
        float * sBestCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
        
        if(threadIdx.x < 1) {
            float d = spanOfAabb(&rootBox, 0);
// first is the best
            if(sBestCost[1] < sBestCost[0] 
                && spanOfAabb(&rootBox, 1) > d * .25f) {
                sBestBin[0] = sBestBin[1];
                d = spanOfAabb(&rootBox, 1);
            }
            
            if(sBestCost[2] < sBestCost[0]
                && spanOfAabb(&rootBox, 2) > d * .25f)
                sBestBin[0] = sBestBin[2];
        }
    }
    
template<int NumBins, int NumThreads, int Dimension>
    __device__ void computeBestBinPerDimension(DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {        
        if((root.y - root.x) < 16)
            computeBinsPerDimensionPrimitive<16, Dimension>(root.y - root.x + 1,
                                    data,
                                    smem,
                                    root,
                                    rootBox);
        else
            computeBinsPerDimensionBatched<NumBins, NumThreads, Dimension>(data,
                                    smem,
                                    root,
                                    rootBox);
    }
 
template<int NumBins, int Dimension>  
    __device__ void computeBinsPerDimensionPrimitive(int numPrimitives,
                                    DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {
/*
 *    layout of memory in int
 *    n  as num bins and num primitives
 *    16 as size of bin
 *
 *    0                                          workId
 *    1               -> 1+3*16-1                best bin per dimension
 *    1+3*16          -> 1+3*16+3-1              cost of best bin per dimension
 *    1+3*16+3        -> 1+3*16+3+n*16-1         bins
 *    1+3*16+3+n*16   -> 1+3*16+3+n*16+n-1       costs
 *    1+3*16+3+n*16+n -> 1+3*16+3+n*16+n+n*n-1   sides
 *
 *    when n = 16
 *    total shared memory 2320 bytes
 *
 */ 
        SplitBin * sBestBin = (SplitBin *)&smem[1];
        float * sBestCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
        SplitBin * sBin = (SplitBin *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3];
        float * sCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * SIZE_OF_SPLITBIN_IN_INT];
        int * sSide = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + NumBins];
        Aabb * sBox = (Aabb *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + NumBins
                                        + NumBins * NumBins];
                                        
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
        int * sideHorizontal = &sSide[threadIdx.x];                      
        
        KeyValuePair * primitiveIndirections = data.primitiveIndirections;
        Aabb * primitiveAabbs = data.primitiveAabbs;
        if(threadIdx.x < NumBins)
            resetSplitBin(sBin[threadIdx.x]);
        
        if(threadIdx.x < numPrimitives) {
// primitive high as split plane             
            sBox[threadIdx.x] = primitiveAabbs[primitiveIndirections[root.x + threadIdx.x].value];
            sBin[threadIdx.x].plane = highPlaneOfAabb(&sBox[threadIdx.x], Dimension);
        }
        
        __syncthreads();
        
/*
 *   layout of binning threads
 *   n as num bins 
 *   m as max num primitives, where m equels n
 *
 *   0      1        2               m-1
 *   m      m+1      m+2             2m-1
 *   2m     2m+1     2m+2            3m-1
 *
 *   (n-1)m (n-1)m+1 (n-1)m+2        nm-1
 *
 *   horizontal i as primitives
 *   vertical   j as bins
 */             
        int j = threadIdx.x / NumBins;
        int i = threadIdx.x - j * NumBins;
        
        if(i < numPrimitives && j < numPrimitives) {
            sSide[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], Dimension) > sBin[j].plane);
        }
    
        __syncthreads();
    
        if(threadIdx.x < numPrimitives) {
            for(i=0; i<numPrimitives; i++) {
                updateSplitBinSide(sBin[threadIdx.x], sBox[i], 
                                    sideHorizontal[i * NumBins]);
            }
        }

        __syncthreads();
        
        
        if(threadIdx.x < numPrimitives) {
            float rootArea = areaOfAabb(&rootBox);
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
        }
        
        __syncthreads();
           
        if(threadIdx.x < 1) {
            int bestI = 0;
            float lowestCost = sCost[0];
            for(i=1; i< numPrimitives; i++) {
                if(lowestCost > sCost[i]) {
                    lowestCost = sCost[i];
                    bestI = i;
                }
            }
            sBestBin[Dimension] = sBin[bestI];
            sBestBin[Dimension].dimension = Dimension;
// store cost here
            sBestCost[Dimension] = sCost[bestI];
        }
        
        __syncthreads();
    }
    
    __device__ int numBinningBatches(int2 range, int batchSize)
    {
        int nbatch = (range.y - range.x + 1)/batchSize;
        if((( range.y - range.x + 1) & (batchSize-1)) > 0) nbatch++;
        return nbatch;
    }
    
    template <int NumBins>
    __device__ void collectBinsBatched(SplitBin & dst,
                                    KeyValuePair * primitiveIndirections,
                                    Aabb * primitiveAabbs,
                                    int * sideHorizontal,
                                    int batchSize,
                                    int begin,
                                    int end)
    {
        for(int i=0; i<batchSize; i++) {
            int j = begin + i;
            if(j<=end) {
                Aabb fBox = primitiveAabbs[primitiveIndirections[j].value];
            
                updateSplitBinSide(dst, fBox, 
                    sideHorizontal[i * NumBins]);
            }
        }
    }
    
    template<int NumBins, int NumThreads, int Dimension>  
    __device__ void computeBinsPerDimensionBatched(DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {
/*
 *    layout of memory in int
 *    n  as num bins
 *    t  as num threads
 *    16 as size of bin
 *    32 as warp size
 *
 *    0                                          workId
 *    1                   -> 1+3*16-1                best bin per dimension
 *    1+3*16              -> 1+3*16+3-1                 cost of best bin per dimension
 *    1+3*16+3            -> 1+3*16+3+n*(t/32)*16-1               bins
 *    1+3*16+3+n*(t/32)*16       -> 1+3*16+3+n*(t/32)*16+n-1               costs
 *    1+3*16+3+n*(t/32)*16+n     -> 1+3*16+3+n*(t/32)*16+n+n*t-1             sides
 *    1+3*16+3+n*(t/32)*16+n+n*t -> 1+3*16+3+n*(t/32)*16+n+n*t+nb*6-1    boxes
 *
 *    when n = 8, t = 256
 *    total shared memory 12528 bytes
 */         
        SplitBin * sBestBin = (SplitBin *)&smem[1];
        float * sBestCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
        SplitBin * sBin = (SplitBin *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3];
        const int numWarps = NumThreads>>5;
        float * sCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * numWarps * SIZE_OF_SPLITBIN_IN_INT];
        int * sSide = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * numWarps * SIZE_OF_SPLITBIN_IN_INT
                                        + NumBins];
        Aabb * sBox = (Aabb *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + NumBins * numWarps * SIZE_OF_SPLITBIN_IN_INT
                                        + NumBins
                                        + NumBins * NumThreads];
                                        
/*
 *    layout of sides
 *    0    n     2n    3n
 *    1    n+1   2n+1  3n+1
 *   
 *    n-1  2n-1  3n-1  4n-1
 *
 *    vertical computeSides
 *    horizonal collectBins
 *
 */
        // int * sideVertical = &sSide[NumBins * threadIdx.x];
        int * sideHorizontal = &sSide[threadIdx.x];
        
       // const int numWarpBins = numWarps * NumBins;
        
        if(threadIdx.x < NumBins) {
            resetSplitBin(sBin[threadIdx.x]);
            sBin[threadIdx.x].plane = binSplitPlane<Dimension>(&rootBox,
                                         NumBins,
                                         threadIdx.x);
        }
        __syncthreads();
        
        KeyValuePair * primitiveIndirections = data.primitiveIndirections;
        Aabb * primitiveAabbs = data.primitiveAabbs;
                    
        const int batchSize = NumThreads / NumBins;
        const int nbatch = numBinningBatches(root, batchSize);
/*
 *    layout of binning threads
 *    n as num bins
 *    m as num threads/n
 *
 *    0      1        2               m-1
 *    m      m+1      m+2             2m-1
 *    2m     2m+1     2m+2            3m-1
 *
 *    (n-1)m (n-1)m+1 (n-1)m+2        nm-1
 *   
 *    horizontal i as primitives
 *    vertical   j as bins
 */
        int j = threadIdx.x / batchSize;
        int i = threadIdx.x - j * batchSize;
        int k, ind;
        
        for(k=0;k<nbatch;k++) {
            ind = root.x + k * batchSize + i;
               
            if(threadIdx.x < batchSize) {
               if(ind <= root.y) {
                   sBox[threadIdx.x] = primitiveAabbs[primitiveIndirections[ind].value];
               }
            }
            __syncthreads();
            
            if(ind <= root.y)
                sSide[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], Dimension) > sBin[j].plane);
        
            __syncthreads();
/*
 *    layout of warp bins
 *    n as num bins
 *    w as num warps (256/32)
 *
 *    0     1        2          w-1         w  warps
 *
 *    0     32n      64n       (w-1)32n
 *    1     32n+1    64n+1     (w-1)32n+1
 *    
 *    n-1   64n-1    96n-1      w32n
 *
 */
            /*
            if(threadIdx.x < numWarpBins) {
                  int w = threadIdx.x / NumBins;
                  int b = threadIdx.x - w * NumBins;
                  
                collectBinsInWarp<NumBins>(sBin[threadIdx.x], 
                    primitiveIndirections,
                    primitiveAabbs,
                    &sSide[b + w * 32 * NumBins],
                    root.x + w * 32 + i * NumThreads,
                    root.y);
            }
            
            __syncthreads();
            
            if(threadIdx.x < NumBins) {
                for(j=1; j<numWarps; j++)
                    combineSplitBin(sBin[threadIdx.x],
                                    sBin[threadIdx.x + j * NumBins]);
                  
            }
            
            __syncthreads();
            */
        
            if(threadIdx.x < NumBins) {  
                collectBinsBatched<NumBins>(sBin[threadIdx.x],
                    primitiveIndirections,
                    primitiveAabbs,
                    sideHorizontal,
                    batchSize,
                    root.x + k * batchSize,
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
            for(i=1; i< NumBins; i++) {
                if(lowestCost > sCost[i]) {
                    lowestCost = sCost[i];
                    bestI = i;
                }
            }
            sBestBin[Dimension] = sBin[bestI];
            sBestBin[Dimension].dimension = Dimension;
// store cost here
            sBestCost[Dimension] = sCost[bestI];
        }
        
        __syncthreads();
    }
    
    template<int NumBins, int NumThreads>
    __device__ void rearrange(DataInterface data, int * smem)
    {
        int iRoot = smem[0];
        int2 root = data.nodes[iRoot];
        int nbatch = numBatches<NumThreads>(root);
        if(nbatch>1) 
            rearrangeBatched<NumBins, NumThreads>(root, nbatch, data, smem);
        else 
            rearrangeInBlock<NumBins, NumThreads>(root, data, smem);
    }
    
    template<int NumBins, int NumThreads>
    __device__ void rearrangeInBlock(int2 root, DataInterface data, int * smem)   
    {
/*
 *    layout of memory in int
 *    t  as num threads
 *    16 as size of bin
 *
 *    0                                          workId
 *    1                -> 1+3*16-1               split bin
 *    1+3*16           -> 1+3*16+t*3-1           sides
 *    1+3*16+t*2       -> 1+3*16+t*3+2-1         group begin
 *    1+3*16+t*2+3     -> 1+3*16+t*3+2+t*4-1     offsets
 *    1+3*16+t*2+3+t*2 -> 1+3*16+t*3+2+t*4+t*2-1 backup indirection
 *    
 */         
        KeyValuePair * major = data.primitiveIndirections;
        KeyValuePair * backup = (KeyValuePair *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 3
                                        + NumThreads * 2];
        Aabb * boxes = data.primitiveAabbs;
        
        const int j = root.x + threadIdx.x;
            
        if(j <= root.y) 
            backup[threadIdx.x] = major[j];
        
        __syncthreads();
        
        SplitBin * sSplit = (SplitBin *)&smem[1];
        float splitPlane = sSplit->plane;
        int splitDimension = sSplit->dimension;
        
        int * groupBegin = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT
                                    + NumThreads * 2];
                                    
        if(threadIdx.x == 0)
            groupBegin[threadIdx.x] = root.x;
            
        if(threadIdx.x == 1)
            groupBegin[threadIdx.x] = root.x + sSplit->leftCount;
        
        __syncthreads();
        
        int * sSide = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
        int * sOffset = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 3];
            
        int * sideVertical = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT
                                        + threadIdx.x * 2];
                                        
        int * offsetVertical = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 3
                                        + threadIdx.x * 2];
                                        
        int splitSide, ind;

        sideVertical[0] = 0;
        sideVertical[1] = 0;
        
        __syncthreads();
        
        if(j<= root.y) {
            splitSide = (lowPlaneOfAabb(&boxes[backup[threadIdx.x].value], splitDimension) > splitPlane);
            sideVertical[splitSide]++;
        }
            
        __syncthreads();
            
        onebitsort::scanInBlock<int>(sOffset, sSide);
            
        if(j<= root.y) {
            ind = groupBegin[splitSide] + offsetVertical[splitSide];
            major[ind] = backup[threadIdx.x];
// for debug purpose only
            // major[ind].key = splitSide;
        }
    }

    template<int NumBins, int NumThreads>
    __device__ void rearrangeBatched(int2 root, int nbatch, DataInterface data, int * smem)   
    {

        KeyValuePair * major = data.primitiveIndirections;
        KeyValuePair * backup = data.intermediateIndirections;
        Aabb * boxes = data.primitiveAabbs;
        
        int i=0;
        for(;i<nbatch;i++)
            writeIndirection(backup, major, root.x + i*NumThreads, root.y);
        
        __syncthreads();
        
/*
 *    layout of memory in int
 *    t  as num threads
 *    16 as size of bin
 *
 *    0                                      workId
 *    1             -> 1+1*16-1              split bin
 *    1+1*16        -> 1+1*16+t*2-1          sides
 *    1+1*16+t*2    -> 1+1*16+t*2+2-1        group begin
 *    1+1*16+t*2+2  -> 1+1*16+t*2+2+t*2-1    offsets
 *
 *    layout of sides
 *
 *    0      1      2        n-1     thread 
 * 
 *    2*0    2*1    2*2      2*(n-1)
 *    2*1-1  2*2-1  2*3-1    2*n-1
 */        
 
        SplitBin * sSplit = (SplitBin *)&smem[1];
        float splitPlane = sSplit->plane;
        int splitDimension = sSplit->dimension;
        
        int * groupBegin = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                    + NumThreads * 2];
                                    
        if(threadIdx.x == 0)
            groupBegin[threadIdx.x] = root.x;
            
        if(threadIdx.x == 1)
            groupBegin[threadIdx.x] = root.x + sSplit->leftCount;
        
        __syncthreads();
        
        int * sSide = &smem[1 + SIZE_OF_SPLITBIN_IN_INT];
        int * sOffset = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 2];
            
        int * sideVertical = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                        + threadIdx.x * 2];
                                        
        int * offsetVertical = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 2
                                        + threadIdx.x * 2];
                                        
        int * sideHorizontal = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                        + threadIdx.x];
                                        
        int * offsetHorizontal = &smem[1 + SIZE_OF_SPLITBIN_IN_INT
                                        + NumThreads * 2
                                        + 2
                                        + threadIdx.x];
        int j, splitSide, ind;
        for(i=0;i<nbatch;i++) {
            sideVertical[0] = 0;
            sideVertical[1] = 0;
            
            __syncthreads();
            
            j = root.x + i*NumThreads + threadIdx.x;
            if(j<= root.y) {
                splitSide = (lowPlaneOfAabb(&boxes[backup[j].value], splitDimension) > splitPlane);
                sideVertical[splitSide]++;
            }
            
            __syncthreads();
            
            onebitsort::scanInBlock<int>(sOffset, sSide);
            
            if(j<= root.y) {
                ind = groupBegin[splitSide] + offsetVertical[splitSide];
                major[ind] = backup[j];
// for debug purpose only
                // major[ind].key = splitSide;
            }
            __syncthreads();
            
            if(threadIdx.x < 2) {
                groupBegin[threadIdx.x] += sideHorizontal[2*(NumThreads-1)]
                                        + offsetHorizontal[2*(NumThreads-1)];
            }
            __syncthreads();
        }
        
    }
    
    template <typename QueueType>
    __device__ void spawn(QueueType * q, DataInterface data, int * smem)
    {
        int & iRoot = smem[0];
        int2 root = data.nodes[iRoot];
        
        SplitBin * sBestBin = (SplitBin *)&smem[1];
        int headToSecond = root.x + sBestBin->leftCount;
        
        const int leftChild = q->enqueue2();
        
        data.nodes[leftChild].x = root.x;
        data.nodes[leftChild].y = headToSecond - 1;
        data.nodeAabbs[leftChild] = sBestBin->leftBox;
        q->releaseTask(leftChild);
        
        const int rightChild = leftChild + 1;
        
        data.nodes[rightChild].x = headToSecond;
        data.nodes[rightChild].y = root.y;
        data.nodeAabbs[rightChild] = sBestBin->rightBox;
        q->releaseTask(rightChild);
        
        int2 childInd;
        childInd.x = (leftChild | 0x80000000);
        childInd.y = (rightChild | 0x80000000);
        data.nodes[iRoot] = childInd;
        
        data.nodeParents[leftChild] = iRoot;
        data.nodeParents[rightChild] = iRoot;
        
        const int level = data.nodeLevels[iRoot] + 1; 
        data.nodeLevels[leftChild] = level;
        data.nodeLevels[rightChild] = level;
    }
    
    template<int NumThreads>
    __device__ int numBatches(int2 range)
    {
        int nbatch = (range.y - range.x + 1)/NumThreads;
        if((( range.y - range.x + 1) & (NumThreads-1)) > 0) nbatch++;
        return nbatch;
    }
    
    __device__ void writeIndirection(KeyValuePair * dst,
                            KeyValuePair * src,
                            int begin, int end)
    {
        int j = begin + threadIdx.x;
        if(j<= end) {
            dst[j] = src[j];
            src[j].key = 9999997;
        }
    }
    
    __device__ void updateSplitBinSide(SplitBin & dst,
                                        Aabb & fBox,
                                        int side)
    {
        if(side) {
            dst.rightCount++;
            expandAabb(dst.rightBox, fBox);
        }
        else {
            dst.leftCount++;
            expandAabb(dst.leftBox, fBox);
        }
    }
    
    template<int Dimension>
    __device__ float binSplitPlane(Aabb * rootBox,
                        uint n,
                        uint ind)
    {
        float d = spanOfAabb(rootBox, Dimension);
        return (lowPlaneOfAabb(rootBox, Dimension) 
                     + (d / (float)n) * ((float)ind + .5f));
    }
    
    __device__ float costOfSplit(SplitBin * bin,
                        float rootBoxArea)
    {
// empty side is invalid
        if(bin->leftCount < 1 || bin->rightCount < 1) return 1e10f;
        
        float leftArea = areaOfAabb(&bin->leftBox);
        float rightArea = areaOfAabb(&bin->rightBox);
    
        return (leftArea / rootBoxArea * (float)bin->leftCount 
                + rightArea / rootBoxArea * (float)bin->rightCount);   
    }
    
    __device__ void resetSplitBin(SplitBin & b)
    {
        resetAabb(b.leftBox);
        resetAabb(b.rightBox);
        b.leftCount = 0;
        b.rightCount = 0;
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
        
        __threadfence_block();
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
