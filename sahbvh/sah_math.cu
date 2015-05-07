#ifndef SAH_MATH_CU
#define SAH_MATH_CU

#include "sahbvh_implement.h"
#include <bvh_math.cu>

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
#endif        //  #ifndef SAH_MATH_CU

