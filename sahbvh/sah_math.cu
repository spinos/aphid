#ifndef SAH_MATH_CU
#define SAH_MATH_CU

#include "sahbvh_implement.h"
#include <bvh_math.cu>

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

inline __device__ void updateSplitBinSide(SplitBin & dst,
                                        Aabb & fBox,
                                        float3 p,
                                        float h,
                                        int side)
{
    if(side) {
        dst.rightCount++;
        expandBinBox(dst.rightBox, fBox, p, h);
    }
    else {
        dst.leftCount++;
        expandBinBox(dst.leftBox, fBox, p, h);
    }
}

inline __device__ void resetSplitBin(SplitBin & b)
{
    b.leftBox.low.x = 2000;
    b.leftBox.low.y = 2000;
    b.leftBox.low.z = 2000;
    b.leftBox.high.x = -2000;
    b.leftBox.high.y = -2000;
    b.leftBox.high.z = -2000;
    b.rightBox.low.x = 2000;
    b.rightBox.low.y = 2000;
    b.rightBox.low.z = 2000;
    b.rightBox.high.x = -2000;
    b.rightBox.high.y = -2000;
    b.rightBox.high.z = -2000;
    b.leftCount = 0;
    b.rightCount = 0;
    b.cost = 0.f;
}

inline __device__ void updateSplitBin(SplitBin & dst,
                                        SplitBin & src)
{
    atomicAdd(&dst.leftCount, src.leftCount);
    atomicAdd(&dst.rightCount, src.rightCount);
    atomicMin(&dst.leftBox.low.x, src.leftBox.low.x);
    atomicMin(&dst.leftBox.low.y, src.leftBox.low.y);
    atomicMin(&dst.leftBox.low.z, src.leftBox.low.z);
    atomicMax(&dst.leftBox.high.x, src.leftBox.high.x);
    atomicMax(&dst.leftBox.high.y, src.leftBox.high.y);
    atomicMax(&dst.leftBox.high.z, src.leftBox.high.z);
    atomicMin(&dst.rightBox.low.x, src.rightBox.low.x);
    atomicMin(&dst.rightBox.low.y, src.rightBox.low.y);
    atomicMin(&dst.rightBox.low.z, src.rightBox.low.z);
    atomicMax(&dst.rightBox.high.x, src.rightBox.high.x);
    atomicMax(&dst.rightBox.high.y, src.rightBox.high.y);
    atomicMax(&dst.rightBox.high.z, src.rightBox.high.z);
}

inline __device__ float splitPlaneOfBin(Aabb * rootBox,
                        uint n,
                        uint ind)
{
    int dimension = ind / n;
    float d = spanOfAabb(rootBox, dimension);
    return (float3_component(rootBox->low, dimension) 
        + (d / (float)n) * ((float)(ind - dimension*n) + .5f));
}

inline __device__ void setSplitSide(int * side,
                        int val,
                        int n)
{
    for(int i=0;i<=n;i++) side[i] = val;
}

inline __device__ int lastBinSplitToRight(float x,
                        float low,
                        float h)
{
    int ibin = (x - low) / h;
    if(((x - low) / h - ibin) > 0.5f) return ibin;
    return (ibin - 1);
}

inline __device__ void computeSplitSide(int * side,
                                        uint dimension,
                                        Aabb * rootBox,
                                        uint numBins,
                                        float * p,
                                        float * boxLow)
{
    float h = spanOfAabb(rootBox, dimension) / (float)numBins;
    
    setSplitSide(side, 0, numBins - 1);
    
    int lastRight = lastBinSplitToRight(p[dimension],
                        boxLow[dimension],
                        h);
    
    setSplitSide(side, 1, lastRight);
}

inline __device__ void updateBins(SplitBin * splitBins,
                                uint iEmission,
                                uint primitiveBegin,
                                KeyValuePair * primitiveInd,
                                Aabb * primitiveAabb,
                                int * sideHorizontal,
                                float3 rootBoxLow,
                                float g,
                                uint dimension,
                                uint nThreads,
                                uint numBins,
                                uint numClusters)
{
    SplitBin & obin = splitBins[iEmission * numBins * 3 
                                + numBins * dimension 
                                + threadIdx.x];
    SplitBin aBin;
    resetSplitBin(aBin);
    
    Aabb fBox;
    uint ind;
    for(int i=0; i<nThreads; i++) {
        ind = primitiveBegin + i;
        if(ind == numClusters) {
            updateSplitBin(obin, aBin);
            break;
        }
        
        fBox = primitiveAabb[primitiveInd[ind].value];
        
        updateSplitBinSide(aBin, fBox, rootBoxLow, g, 
            sideHorizontal[i*SAH_MAX_NUM_BINS]);
        
        if(i == nThreads-1) {
            updateSplitBin(obin, aBin);
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

inline __device__ float costOfSplit(SplitBin * bin,
                        float rootBoxArea,
                        float h)
{
// empty side is invalid
    if(bin->leftCount < 1 || bin->rightCount < 1) return 1e10f;
    
    float leftArea = areaOfBinBox(&bin->leftBox, h);
    float rightArea = areaOfBinBox(&bin->rightBox, h);

    return (leftArea / rootBoxArea * (float)bin->leftCount 
            + rightArea / rootBoxArea * (float)bin->rightCount);   
}

#endif        //  #ifndef SAH_MATH_CU

