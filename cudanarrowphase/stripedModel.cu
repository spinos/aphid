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
    
    uint4 r;
    r.x = pointStarts[objI] + indices[indexStarts[objI] + elmI].x;
    r.y = pointStarts[objI] + indices[indexStarts[objI] + elmI].y;
    r.z = pointStarts[objI] + indices[indexStarts[objI] + elmI].z;
    r.w = pointStarts[objI] + indices[indexStarts[objI] + elmI].w;
    return r;
}
#endif        //  #ifndef STRIPEDMODEL_CU

