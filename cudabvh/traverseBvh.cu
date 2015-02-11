#include "traverseBvh_implement.h"
#include <radixsort_implement.h>
#include <bvh_math.cu>
#include <CudaBase.h>

#define B3_PLVBH_TRAVERSE_MAX_STACK_SIZE 128

__device__ int rayIntersectsAabb(float3 rayOrigin, float rayLength, 
                                float3 rayNormalizedDirection, 
                                const Aabb & aabb,
                                float & distanceMin, float & distanceMax)
{
	//AABB is considered as 3 pairs of 2 planes( {x_min, x_max}, {y_min, y_max}, {z_min, z_max} ).
	//t_min is the point of intersection with the closer plane, t_max is the point of intersection with the farther plane.
	//
	//if (rayNormalizedDirection.x < 0.0f), then max.x will be the near plane 
	//and min.x will be the far plane; otherwise, it is reversed.
	//
	//In order for there to be a collision, the t_min and t_max of each pair must overlap.
	//This can be tested for by selecting the highest t_min and lowest t_max and comparing them.
	
	//int3 isNegative = isless( rayNormalizedDirection, make_float3(0.0f, 0.0f, 0.0f) );	//isless(x,y) returns (x < y)
	int3 isNegative = make_int3(rayNormalizedDirection.x < 0.f, rayNormalizedDirection.y < 0.f, rayNormalizedDirection.z < 0.f);
	//When using vector types, the select() function checks the most signficant bit, 
	//but isless() sets the least significant bit.
	//isNegative <<= 31;

	//select(b, a, condition) == condition ? a : b
	//When using select() with vector types, (condition[i]) is true if its most significant bit is 1
	float3 t_min = float3_difference( select(aabb.high, aabb.low, isNegative), rayOrigin );
	//divide_float3(t_min, rayNormalizedDirection);
	float3 t_max = float3_difference( select(aabb.low, aabb.high, isNegative), rayOrigin );
	//divide_float3(t_max, rayNormalizedDirection);
	
	t_min.x /= rayNormalizedDirection.x;
	t_min.y /= rayNormalizedDirection.y;
	t_min.z /= rayNormalizedDirection.z;
	
	t_max.x /= rayNormalizedDirection.x;
	t_max.y /= rayNormalizedDirection.y;
	t_max.z /= rayNormalizedDirection.z;
	
	//float t_min_final = 0.0f;
	//float t_max_final = rayLength;
	
	distanceMin = 0.f; 
	distanceMax = rayLength;

	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	distanceMin = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, distanceMin)) );
	distanceMax = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, distanceMax)) );

	return (distanceMin <= distanceMax);
}

__global__ void rayTraverseIterative_kernel(RayInfo * rays,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,								
								uint numRays)
{
	uint rayIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(rayIndex >= numRays) return;
	
	float3 rayFrom = rays[rayIndex].origin;
	float3 rayTo = rays[rayIndex].destiny;
	float3 rayNormalizedDirection = float3_difference(rayTo, rayFrom);
	
	float rayLength = float3_length(rayNormalizedDirection);
	rayNormalizedDirection = scale_float3_by(rayNormalizedDirection, 1.0 / rayLength);

	uint stack[B3_PLVBH_TRAVERSE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
		
	float minHitDistanct = HUGE_VALUE;

	float2 testDistance;
	int isLeaf;
	
	while(stackSize > 0)
	{
		uint internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);

		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];

		if( rayIntersectsAabb(rayFrom, rayLength, rayNormalizedDirection, bvhNodeAabb, testDistance.x, testDistance.y)  )
		{    
		    
			if(isLeaf)
			{
			    if(testDistance.y < minHitDistanct) minHitDistanct = testDistance.y;
			    //if( rayIntersectsAabb(rayFrom, rayLength, rayNormalizedDirection, bvhNodeAabb, testDistance)  )
			     //minHitDistanct = 32.f;
				// int2 rayRigidPair;
				// rayRigidPair.x = rayIndex;
				// rayRigidPair.y = rigidAabbs[bvhRigidIndex].m_minIndices[3];
				// 
				// int pairIndex = atomic_inc(out_numRayRigidPairs);
				// if(pairIndex < maxRayRigidPairs) out_rayRigidPairs[pairIndex] = rayRigidPair;
				// 
				
			}
			else {
			

				// if(stackSize + 2 > B3_PLVBH_TRAVERSE_MAX_STACK_SIZE)
				// {
					// //Error
					// break;
				// }
				// else
				// {
				    stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
					stackSize++;
					stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
					stackSize++;
				// }
			}
		}
		
	}
	
	if(minHitDistanct < HUGE_VALUE) {
	    rayNormalizedDirection = scale_float3_by(rayNormalizedDirection, minHitDistanct);
	}
	
	rays[rayIndex].destiny = float3_add(rays[rayIndex].origin, rayNormalizedDirection);
}

__global__ void testRay_kernel(RayInfo * o_ray, Aabb box, float3 pfrom, float h, uint width, uint n)
{
    uint rayIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(rayIndex >= n) return;
	
	o_ray[rayIndex].origin = pfrom;
	
	uint v = rayIndex / width;
	uint u = rayIndex - v * width;
	float hwidth = h * width * 0.5f;
	
	float3 d = make_float3(h * u - hwidth, -hwidth, h * v - hwidth);
	d = float3_normalize(d);
	
	o_ray[rayIndex].destiny = make_float3(pfrom.x + d.x * 1000.f, pfrom.y + d.y * 1000.f, pfrom.z + d.z * 1000.f);
}

extern "C" void bvhRayTraverseIterative(RayInfo * rays,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,								
								uint numRays)
{
    int tpb = CudaBase::LimitNThreadPerBlock(24, 40);
    
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numRays, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    rayTraverseIterative_kernel<<< grid, block >>>(rays,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,								
								numRays);
}

extern "C" void bvhTestRay(RayInfo * o_rays, float3 origin, float boxSize, uint w, uint n)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(n, 512);
    
    dim3 grid(nblk, 1, 1);
    
    float h = 2.f;
    
    Aabb box;
    box.low.x = -boxSize + 1; box.low.y = 0.f; box.low.z = -boxSize;
    box.high.x = boxSize + 1; box.high.y = boxSize; box.high.z = boxSize;
    testRay_kernel<<< grid, block >>>(o_rays, box, origin, h, w, n);
}
