#ifndef STRIPEDMODEL_CUH
#define STRIPEDMODEL_CUH

#include "bvh_math.cuh"

// limit num elements per object to 1^20-1
inline __device__ uint combineObjectElementInd(uint objectIdx, uint elementIdx)
{ return (objectIdx<<20 | elementIdx); }

inline __device__ uint extractObjectInd(uint combined)
{ return (combined>>20);}

inline __device__ uint extractElementInd(uint combined)
{ return ((combined<<12)>>12);}

inline __device__ uint4 computePointIndex(uint * pointStarts,
                                            uint * indexStarts,
                                            uint4 * indices,
                                            uint combined)
{
    const uint objI = extractObjectInd(combined);
    const uint elmI = extractElementInd(combined);
    const uint & objectDrift = pointStarts[objI];
    const uint4 & elementInd = indices[indexStarts[objI] + elmI];
    uint4 r;
    r.x = objectDrift + elementInd.x;
    r.y = objectDrift + elementInd.y;
    r.z = objectDrift + elementInd.z;
    r.w = objectDrift + elementInd.w;
    return r;
}
#endif        //  #ifndef STRIPEDMODEL_CUH

