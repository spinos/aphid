#ifndef CUFEMTETRAHEDRON_IMPLEMENT_H
#define CUFEMTETRAHEDRON_IMPLEMENT_H

#include "bvh_common.h"
#include <radixsort_implement.h>
extern "C" {
    
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd);

void cuFemTetrahedron_calculateRe(mat33 * dst, 
                                    float3 * pos, 
                                    float3 * pos0,
                                    uint4 * indices,
                                    uint maxInd);

void cuFemTetrahedron_resetStiffnessMatrix(mat33 * dst,
                                    uint maxInd);

void cuFemTetrahedron_stiffnessAssembly(mat33 * dst,
                                        float3 * pos,
                                        uint4 * vert,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        uint maxBufferInd,
                                        uint maxInd);

void cuFemTetrahedron_resetForce(float3 * dst,
                                    uint maxInd);

void cuFemTetrahedron_internalForce(float3 * dst,
                                    float3 * pos,
                                    uint4 * tetvert,
                                    mat33 * orientation,
                                    KeyValuePair * tetraInd,
                                    uint * bufferIndices,
                                    uint maxBufferInd,
                                    uint maxInd);

void cuFemTetrahedron_computeRhsA(float3 * rhs,
                                float3 * pos,
                                float3 * vel,
                                float * mass,
                                mat33 * stiffness,
                                uint * rowPtr,
                                uint * colInd,
                                float3 * f0,
                                float3 * externalForce,
                                float dt,
                                uint maxInd);

void cuFemTetrahedron_externalForce(float3 * dst,
                                float * mass,
                                uint maxInd);
								
void cuFemTetrahedron_integrate(float3 * pos, 
								float3 * vel, 
								uint * anchor,
								float dt, 
								uint maxInd);
}
#endif        //  #ifndef CUFEMTETRAHEDRON_IMPLEMENT_H

