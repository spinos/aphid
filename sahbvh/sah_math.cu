#ifndef SAH_MATH_CU
#define SAH_MATH_CU

#include "sahbvh_implement.h"
#include <bvh_math.cu>

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
                        uint dimension,
                        uint n,
                        uint ind)
{
    float d = spanOfAabb(rootBox, dimension);
    float * ll = &(rootBox->low.x);
    float fmn = ll[dimension];
    float h = d / (float)n; 
    return fmn + h * (float)ind + h * .5f;
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
                                Aabb * clusterAabbs,
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
        
        fBox = clusterAabbs[ind];
        
        updateSplitBinSide(aBin, fBox, rootBoxLow, g, 
            sideHorizontal[i*SAH_MAX_NUM_BINS]);
        
        if(i == nThreads-1) {
            updateSplitBin(obin, aBin);
        }
    }
    
}

#endif        //  #ifndef SAH_MATH_CU

