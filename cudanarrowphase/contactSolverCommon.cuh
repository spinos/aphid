#include "simpleContactSolver_implement.h"
#include "bvh_math.cuh"
#include "barycentric.cuh"
#include "stripedModel.cuh"
#define SETCONSTRAINT_TPB 128
#define VYACCELERATION 0.1635f // 9.81 / 60
#define SOLVECONTACT_TPB 256

inline __device__ float computeRelativeVelocity1(float3 nA,
                            float3 nB,
                            float3 linearVelocityA, 
                            float3 linearVelocityB)
{
    return float3_dot(linearVelocityA, nA) +
            float3_dot(linearVelocityB, nB);
}
