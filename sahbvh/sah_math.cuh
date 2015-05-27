#include "sah_common.h"
#include "bvh_math.cuh"
inline __device__ void resetSplitBin(SplitBin & b)
{
    resetAabb(b.leftBox);
    resetAabb(b.rightBox);
    b.leftCount = 0;
    b.rightCount = 0;
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
                                        float p,
                                        float boxLow)
{
    float h = spanOfAabb(rootBox, dimension) / (float)numBins;
    
    setSplitSide(side, 0, numBins - 1);
    
    int lastRight = lastBinSplitToRight(p,
                        boxLow,
                        h);
    
    setSplitSide(side, 1, lastRight);
}

inline __device__ void updateSplitBinSide(SplitBin & dst,
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

template <int NumBins, int Dimension>
inline __device__ void computeSides(int * sideVertical,
                                    Aabb rootBox,
                                    KeyValuePair * primitiveIndirections,
                                    Aabb * primitiveAabbs,
                                    int begin,
                                    int end)
{
    int j = begin + threadIdx.x;
    if(j<= end) {
        Aabb clusterBox = primitiveAabbs[primitiveIndirections[j].value];
        //float3 center = centroidOfAabb(clusterBox);
        float p = float3_component(clusterBox.low, Dimension);
        float boxLow = float3_component(rootBox.low, Dimension);    
        
        computeSplitSide(sideVertical,
                        Dimension,
                        &rootBox,
                        NumBins,
                        p,
                        boxLow);
    }
}

template <int NumBins, int NumThreads>
inline __device__ void collectBins(SplitBin & dst,
                                KeyValuePair * primitiveIndirections,
                                Aabb * primitiveAabbs,
                                int * sideHorizontal,
                                int begin,
                                int end)
{
    for(int i=0; i<NumThreads; i++) {
        int j = begin + i;
        if(j<=end) {
            Aabb fBox = primitiveAabbs[primitiveIndirections[j].value];
        
            updateSplitBinSide(dst, fBox, 
                sideHorizontal[i * NumBins]);
        }
    }
}

inline __device__ float costOfSplit(SplitBin * bin,
                        float rootBoxArea)
{
// empty side is invalid
    if(bin->leftCount < 1 || bin->rightCount < 1) return 1e10f;
    
    float leftArea = areaOfAabb(&bin->leftBox);
    float rightArea = areaOfAabb(&bin->rightBox);

    return (leftArea / rootBoxArea * (float)bin->leftCount 
            + rightArea / rootBoxArea * (float)bin->rightCount);   
}

template<int Dimension>
inline __device__ float binSplitPlane(Aabb * rootBox,
                        uint n,
                        uint ind)
{
    float d = spanOfAabb(rootBox, Dimension);
    return (float3_component(rootBox->low, Dimension) 
                 + (d / (float)n) * ((float)ind + .5f));
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

inline __device__ void writeIndirection(KeyValuePair * dst,
                            KeyValuePair * src,
                            int begin, int end)
{
    int j = begin + threadIdx.x;
    if(j<= end) {
        dst[j] = src[j];
        src[j].key = 9999997;
    }
}

template <int NumBins>
inline __device__ void collectBinsInWarp(SplitBin & dst,
                                    KeyValuePair * primitiveIndirections,
                                    Aabb * primitiveAabbs,
                                    int * side,
                                    int begin,
                                    int end)
    {
        for(int i=0; i<32; i++) {
            int j = begin + i;
            if(j<=end) {
                Aabb fBox = primitiveAabbs[primitiveIndirections[j].value];
                
                updateSplitBinSide(dst, fBox, 
                  side[i*NumBins]);
            }
        }
    }
    
inline __device__ void combineSplitBin(SplitBin & a, const SplitBin & b)
{
    a.leftCount += b.leftCount;
    a.rightCount += b.leftCount;
    expandAabb(a.leftBox, b.leftBox);
    expandAabb(a.rightBox, b.rightBox);
}
