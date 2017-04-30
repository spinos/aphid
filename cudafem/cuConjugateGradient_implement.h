#ifndef CUCONJUGATEGRADIENT_IMPLEMENT_H
#define CUCONJUGATEGRADIENT_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
void cuConjugateGradient_Ax(float3 * X,
                            float3 * update,
                            float3 * residual,
                            float * d,
                            float * d2,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            uint maxInd);

void cuConjugateGradient_addResidual(float3 * prev,
                            float3 * residual,
                            float d4,
                            uint * fixed,
                            uint maxInd);

void cuConjugateGradient_prevresidual(float3 * prev,
                            float3 * residual,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            float3 * guess,
                            float3 * rhs,
                            uint maxInd);

void cuConjugateGradient_addX(float3 * X,
                            float3 * residual,
                            float * d,
                            float3 * prev,
                            float3 * update,
                            float d3,
                            uint * fixed,
                            uint maxInd);
}
#endif        //  #ifndef CUCONJUGATEGRADIENT_IMPLEMENT_H

