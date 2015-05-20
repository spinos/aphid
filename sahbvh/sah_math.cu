#ifndef SAH_MATH_CU
#define SAH_MATH_CU

#include "sahbvh_implement.h"
#include "sah_math.cuh"

inline __device__ void binAabbToAabb(Aabb & dst,
                                BinAabb & src,
                                float3 p,
                                float h)
{
    dst.low.x = p.x + (float)src.low.x * h;
    dst.low.y = p.y + (float)src.low.y * h;
    dst.low.z = p.z + (float)src.low.z * h;
    dst.high.x = p.x + (float)src.high.x * h;
    dst.high.y = p.y + (float)src.high.y * h;
    dst.high.z = p.z + (float)src.high.z * h;
}

inline __device__ void expandBinBox(BinAabb & dst,
                                    Aabb & src,
                                    float3 p,
                                    float h)
{
    float x;
    int r;
    
    x= src.low.x - p.x;
    r= x / h;
    if(dst.low.x > r) dst.low.x = r;
    
    x = src.low.y - p.y;
    r= x / h;
    if(dst.low.y > r) dst.low.y = r;
    
    x = src.low.z - p.z;
    r= x / h;
    if(dst.low.z > r) dst.low.z = r;
    
    x = src.high.x - p.x;
    r = x / h + 1;
    if(dst.high.x < r) dst.high.x = r;
    
    x = src.high.y - p.y;
    r = x / h + 1;
    if(dst.high.y < r) dst.high.y = r;
    
    x = src.high.z - p.z;
    r = x / h + 1;
    if(dst.high.z < r) dst.high.z = r;
}

inline __device__ void updateSplitBin(SplitBin & dst,
                                        SplitBin & src)
{
    dst.leftCount += src.leftCount;
    dst.rightCount += src.rightCount;
    expandAabb(dst.leftBox, src.leftBox);
    expandAabb(dst.rightBox, src.rightBox);
}

inline __device__ void collectBins(SplitBin & dst,
                                KeyValuePair * primitiveInd,
                                Aabb * primitiveAabb,
                                int * sideHorizontal,
                                int begin,
                                int end)
{
    int tid = 0;
    for(int i=begin; i<end; i++) {
        Aabb fBox = primitiveAabb[primitiveInd[i].value];
        
        updateSplitBinSide(dst, fBox, 
            sideHorizontal[tid*SAH_MAX_NUM_BINS]);
        
        tid++;
    }
}

inline __device__ void updateBins(SplitBin * splitBins,
                                uint primitiveBegin,
                                KeyValuePair * primitiveInd,
                                Aabb * primitiveAabb,
                                int * sideHorizontal,
                                uint dimension,
                                uint nThreads,
                                uint numBins,
                                uint numClusters)
{
    SplitBin aBin;
    resetSplitBin(aBin);
    
    Aabb fBox;
    uint ind;
    for(int i=0; i<nThreads; i++) {
        ind = primitiveBegin + i;
        if(ind == numClusters) {
            updateSplitBin(splitBins[threadIdx.x], aBin);
            break;
        }
        
        fBox = primitiveAabb[primitiveInd[ind].value];
        
        updateSplitBinSide(aBin, fBox, 
            sideHorizontal[i*SAH_MAX_NUM_BINS]);
        
        if(i == nThreads-1) {
            updateSplitBin(splitBins[threadIdx.x], aBin);
        }
    }  
}

inline __device__ float areaOfBinBox(BinAabb * box,
                                    float h)
{
    float dx = box->high.x - box->low.x;
    float dy = box->high.y - box->low.y;
    float dz = box->high.z - box->low.z;
    if(dx <= 0.f || dy <= 0.f || dz <= 0.f) return 0.f;
    
    dx *= h;
    dy *= h;
    dz *= h;
    
    return (dx * dy + dy * dz + dz * dx) * 2.f;
}

#endif        //  #ifndef SAH_MATH_CU

