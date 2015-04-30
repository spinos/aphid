#ifndef STRIPEDMODEL_CU
#define STRIPEDMODEL_CU

#include <bvh_math.cu>

inline __device__ uint combineObjectElementInd(uint objectIdx, uint elementIdx)
{ return (objectIdx<<24 | elementIdx); }

inline __device__ uint extractObjectInd(uint combined)
{ return (combined>>24);}

inline __device__ uint extractElementInd(uint combined)
{ return ((combined<<7)>>7);}

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
#endif        //  #ifndef STRIPEDMODEL_CU

