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
        return (sBestCost[0] < (root.y - root.x -.6f));
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
        
            __threadfence_block();
            __syncthreads();

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
        computeBestBin3Dimensions<NumBins, NumThreads>(data,
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
    
    template<int NumBins, int NumThreads>
    __device__ void computeBestBin3Dimensions(DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {      
        float rootArea = areaOfAabb(&rootBox);
        if((root.y - root.x) < 16) {
            computeBinsPrimitive<16>(root.y - root.x + 1,
                                    data,
                                    smem,
                                    root,
                                    rootBox);
            findBestBin3Dimensions<16>(smem, rootArea);
        }
        else {
            computeBinsBatched<NumBins, NumThreads>(data,
                                    smem,
                                    root,
                                    rootBox);
            findBestBin3Dimensions<NumBins>(smem, rootArea);
        }
    }
    
    template<int NumBins>
    __device__ void findBestBin3Dimensions(int * smem, float rootArea)
    {
        SplitBin * sBestBin = (SplitBin *)&smem[1];
        float * sBestCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT];
           
        SplitBin * sBin = (SplitBin *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3];
        
        SplitBin * sBinX = sBin;
        SplitBin * sBinY = &sBin[    NumBins];
        SplitBin * sBinZ = &sBin[2 * NumBins];
        
        float * sCost = (float *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                               + 4 * NumBins * SIZE_OF_SPLITBIN_IN_INT];
        
        float * sCostX = sCost;
        float * sCostY = &sCost[    NumBins];
        float * sCostZ = &sCost[2 * NumBins];
        
        if(threadIdx.x < NumBins * 3) {
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
        }
         __syncthreads();
        
        int i, bestI;
        float lowestCost;
        if(threadIdx.x == 0) {
            bestI = 0;
            lowestCost = sCostX[0];
            for(i=1; i< NumBins; i++) {
                if(lowestCost > sCostX[i]) {
                    lowestCost = sCostX[i];
                    bestI = i;
                }
            }
            sBestBin[0] = sBinX[bestI];
            sBestBin[0].dimension = 0;
            sBestCost[0] = lowestCost;
        }
        
        if(threadIdx.x == 1) {
            bestI = 0;
            lowestCost = sCostY[0];
            for(i=1; i< NumBins; i++) {
                if(lowestCost > sCostY[i]) {
                    lowestCost = sCostY[i];
                    bestI = i;
                }
            }
            sBestBin[1] = sBinY[bestI];
            sBestBin[1].dimension = 1;
            sBestCost[1] = lowestCost;
        }
        
        if(threadIdx.x == 2) {
            bestI = 0;
            lowestCost = sCostZ[0];
            for(i=1; i< NumBins; i++) {
                if(lowestCost > sCostZ[i]) {
                    lowestCost = sCostZ[i];
                    bestI = i;
                }
            }
            sBestBin[2] = sBinZ[bestI];
            sBestBin[2].dimension = 2;
            sBestCost[2] = lowestCost;
        }
        
        __syncthreads();
    }
 
template<int NumBins>  
    __device__ void computeBinsPrimitive(int numPrimitives,
                                    DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {
/*
 *    layout of memory in int
 *    n  as num bins and n >= num primitives
 *    16 as size of bin
 *
 *    0                                          workId
 *    1               -> 1+3*16-1                best bin per dimension
 *    1+3*16          -> 1+3*16+3-1              cost of best bin per dimension
 *    1+3*16+3        -> 1+3*16+3+n*4*16-1       bins for 3 dimensions
 *    1+3*16+3+n*4*16 -> 1+3*16+3+n*4*16+n*4-1   costs for 3 dimensions
 *    1+3*16+3+n*4*16+n*4 -> 1+3*16+3+n*4*16+n*4+n*n*4-1 sides for 3 dimensions
 *    1+3*16+3+n*4*16+n*4+n*n*4 -> 1+3*16+3+n*4*16+n*4+n*n*4+n*6-1    boxes
 *    when n = 16
 *    total shared memory 52+1024+64+1024+96 = 9040 bytes
 *
 */ 
        SplitBin * sBin = (SplitBin *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3];
        
        SplitBin *sBinX = sBin;
        SplitBin *sBinY = &sBin[    NumBins];
        SplitBin *sBinZ = &sBin[2 * NumBins];
                                        
        int * sSide = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + 4 * NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + 4 * NumBins];          
        int * sSideX = sSide;
        int * sSideY = &sSide[    NumBins * NumBins];
        int * sSideZ = &sSide[2 * NumBins * NumBins];
                                        
        Aabb * sBox = (Aabb *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + 4 * NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + 4 * NumBins
                                        + 4 * NumBins * NumBins];   
                                        
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
        
        KeyValuePair * primitiveIndirections = data.primitiveIndirections;
        Aabb * primitiveAabbs = data.primitiveAabbs;
        if(threadIdx.x < 3 * NumBins)
            resetSplitBin(sBin[threadIdx.x]);
        
        if(threadIdx.x < numPrimitives) {
// primitive high as split plane             
            sBox[threadIdx.x] = primitiveAabbs[primitiveIndirections[root.x + threadIdx.x].value];
            sBinX[threadIdx.x].plane = highPlaneOfAabb(&sBox[threadIdx.x], 0);
        }
        __syncthreads();
        
        if(threadIdx.x > 31 && threadIdx.x < 32 + numPrimitives)
            sBinY[threadIdx.x - 32].plane = highPlaneOfAabb(&sBox[threadIdx.x - 32], 1);
        
        if(threadIdx.x > 63 && threadIdx.x < 64 + numPrimitives)
            sBinZ[threadIdx.x - 64].plane = highPlaneOfAabb(&sBox[threadIdx.x - 64], 2);
        
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
            sSideX[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 0) > sBinX[j].plane);
            sSideY[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 1) > sBinY[j].plane);
            sSideZ[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 2) > sBinZ[j].plane);
        }
    
        __syncthreads();
    
        if(threadIdx.x < numPrimitives) {
            for(i=0; i<numPrimitives; i++) {
                updateSplitBinSide(sBinX[threadIdx.x], sBox[i], 
                                    sSideX[threadIdx.x + i * NumBins]);
            }
        }
        
        if(threadIdx.x > 31 && threadIdx.x < 32+numPrimitives) {
            for(i=0; i<numPrimitives; i++) {
                updateSplitBinSide(sBinY[threadIdx.x-32], sBox[i], 
                                    sSideY[threadIdx.x-32 + i * NumBins]);
            }
        }
        
        if(threadIdx.x > 63 && threadIdx.x < 64+numPrimitives) {
            for(i=0; i<numPrimitives; i++) {
                updateSplitBinSide(sBinZ[threadIdx.x-64], sBox[i], 
                                    sSideZ[threadIdx.x-64 + i * NumBins]);
            }
        }

        __syncthreads();
          
    }
    
    __device__ int numBinningBatches(int2 range, int batchSize)
    {
        int nbatch = (range.y - range.x + 1)/batchSize;
        if((( range.y - range.x + 1) & (batchSize-1)) > 0) nbatch++;
        return nbatch;
    }
    
    template <int NumBins, int ThreadOffset>
    __device__ void collectBinsBatched(SplitBin * bins,
                                    Aabb * boxes,
                                    int * sides,
                                    int batchSize,
                                    int begin,
                                    int end)
    {
        for(int i=0; i<batchSize; i++) {
            if((begin + i)<=end) {
                updateSplitBinSide(bins[threadIdx.x - ThreadOffset], boxes[i], 
                    sides[threadIdx.x - ThreadOffset + i * NumBins]);
            }
        }
    }
    
    template<int NumBins, int NumThreads>  
    __device__ void computeBinsBatched(DataInterface data,
                                    int * smem,
                                    int2 root,
                                    Aabb rootBox)
    {
/*
 *    layout of memory in int
 *    n  as num bins
 *    t  as num threads
 *    16 as size of bin
 *    nb is batch size nt/n = 32
 *
 *    0                                          workId
 *    1                   -> 1+3*16-1                best bin per dimension
 *    1+3*16              -> 1+3*16+3-1                 cost of best bin per dimension
 *    1+3*16+3            -> 1+3*16+3+n*4*16-1               bins for 3 dimensions
 *    1+3*16+3+n*4*16       -> 1+3*16+3+n*4*16+n*4-1               costs for 3 dimensions
 *    1+3*16+3+n*4*16+n*4        -> 1+3*16+3+n*4*16+n*4+n*nb*4-1         sides for 3 dimensions
 *    1+3*16+3+n*4*16+n*4+n*nb*4 -> 1+3*16+3+n*4*16+n*4+n*nb*4+nb*6-1    boxes
 *
 *    when n = 8, t = 256
 *    total shared memory 52+512+32+1024+192 = 7248 bytes
 */       
        const int batchSize = NumThreads / NumBins;
        
        SplitBin * sBin = (SplitBin *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3];
        SplitBin *sBinX = sBin;
        SplitBin *sBinY = &sBin[    NumBins];
        SplitBin *sBinZ = &sBin[2 * NumBins];
        int * sSide = &smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + 4 * NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + 4 * NumBins];
        int * sSideX = sSide;
        int * sSideY = &sSide[    NumBins * batchSize];
        int * sSideZ = &sSide[2 * NumBins * batchSize];
        Aabb * sBox = (Aabb *)&smem[1 + 3 * SIZE_OF_SPLITBIN_IN_INT + 3
                                        + 4 * NumBins * SIZE_OF_SPLITBIN_IN_INT
                                        + 4 * NumBins
                                        + 4 * NumBins * batchSize];                                       
/*
 *    layout of sides
 *    0    n    2n    .  31n
 *    1    n+1  2n+1  .  31n+1
 *    .
 *    n-1 2n-1  3n-1  .  32n-1
 *
 *    vertical computeSides
 *    horizonal collectBins
 *
 */ 
        if(threadIdx.x < NumBins) {
            resetSplitBin(sBinX[threadIdx.x]);
            sBinX[threadIdx.x     ].plane = binSplitPlane<NumBins, 0, 0>(&rootBox);
        }
        
        if(threadIdx.x > 31 && threadIdx.x < 32 + NumBins) {
            resetSplitBin(sBinY[threadIdx.x - 32]);
            sBinY[threadIdx.x - 32].plane = binSplitPlane<NumBins, 1, 32>(&rootBox);
        }
        
        if(threadIdx.x > 63 && threadIdx.x < 64 + NumBins) {
            resetSplitBin(sBinZ[threadIdx.x - 64]);
            sBinZ[threadIdx.x - 64].plane = binSplitPlane<NumBins, 2, 64>(&rootBox);
        }
        __syncthreads();
        
        KeyValuePair * primitiveIndirections = data.primitiveIndirections;
        Aabb * primitiveAabbs = data.primitiveAabbs;
                    
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
            
            if(ind <= root.y) {
                sSideX[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 0) > sBinX[j].plane);
                sSideY[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 1) > sBinY[j].plane);
                sSideZ[i*NumBins + j] = (lowPlaneOfAabb(&sBox[i], 2) > sBinZ[j].plane);
            }
        
            __syncthreads();
        
            if(threadIdx.x < NumBins) {  
                collectBinsBatched<NumBins, 0>(sBinX,
                    sBox,
                    sSideX,
                    batchSize,
                    root.x + k * batchSize,
                    root.y);
            }
            
            if(threadIdx.x >31 && threadIdx.x < 32+NumBins) {  
                collectBinsBatched<NumBins, 32>(sBinY,
                    sBox,
                    sSideY,
                    batchSize,
                    root.x + k * batchSize,
                    root.y);
            }
            
            if(threadIdx.x >63 && threadIdx.x < 64+NumBins) {  
                collectBinsBatched<NumBins, 64>(sBinZ,
                    sBox,
                    sSideZ,
                    batchSize,
                    root.x + k * batchSize,
                    root.y);
            }
    
            __syncthreads();
        } 
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
            // src[j].key = 9999997;
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
    
    template<int NumBins, int Dimension, int ThreadOffset>
    __device__ float binSplitPlane(Aabb * rootBox)
    {
        int ind = threadIdx.x - ThreadOffset;
        float d = spanOfAabb(rootBox, Dimension);
        return (lowPlaneOfAabb(rootBox, Dimension) 
                     + (d / (float)NumBins) * ((float)ind + .5f));
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
