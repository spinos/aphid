#ifndef CUREDUCESUM_IMPLEMENT_H
#define CUREDUCESUM_IMPLEMENT_H

#include "reduce_common.h"

extern "C" {
void cuReduce_F_Sum(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads); 

void cuReduce_F_Max(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads); 

void cuReduce_F_Min(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads); 

void cuReduce_F_MinMax1(float2 * dst, float * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_F_MinMax2(float2 * dst, float2 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Min1(float3 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Pnt_Min2(float3 * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Max1(float3 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Pnt_Max2(float3 * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Pnt_Min1(float3 * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Pnt_Max1(float3 * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_MinX(float * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Pnt_MinX(float * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Min1_Flt4(float4 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Min2_Flt4(float4 * dst, float4 * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Max1_Flt4(float4 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads);

void cuReduce_Box_Max2_Flt4(float4 * dst, float4 * src,
                    uint n, uint nBlocks, uint nThreads);
}
#endif        //  #ifndef CUREDUCE_IMPLEMENT_H

