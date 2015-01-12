#include "traverseBvh_implement.h"
#include <radixsort_implement.h>

#define B3_PLVBH_TRAVERSE_MAX_STACK_SIZE 128

__device__ int isLeafNode(int index) 
{ return (index >> 31 == 0); }

__device__ int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }

__device__ float float3_length(float3 v) 
{ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }

__device__ float float3_length2(float3 v) 
{ return (v.x*v.x + v.y*v.y + v.z*v.z); }

__device__ float3 float3_normalize(float3 v)
{
	float l = float3_length(v);
	l = 1.0 / l;
	return make_float3(v.x * l, v.y * l, v.z * l);
}

__device__ float3 float3_difference(float3 v1, float3 v0)
{ return make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z); }

__device__ float3 float3_add(float3 v1, float3 v0)
{ return make_float3(v1.x + v0.x, v1.y + v0.y, v1.z + v0.z); }

__device__ void scale_float3_by(float3 & v, float s)
{ v.x *= s; v.y *= s; v.z *= s; }

__device__ void divide_float3(float3 & v, float3 s)
{ v.x /= s.x; v.y /= s.y; v.z /= s.z; }

__device__ int3 isless(float3 v, float3 threshold)
{ return make_int3(v.x < threshold.x, v.y < threshold.y, v.z < threshold.z); }

__device__ float3 select(float3 a, float3 b, int3 con)
{
    float x = con.x ? a.x : b.x;
    float y = con.y ? a.y : b.y;
    float z = con.z ? a.z : b.z;
    return make_float3(x, y, z);
}

__device__ int rayIntersectsAabb(float3 rayOrigin, float rayLength, 
                                float3 rayNormalizedDirection, 
                                Aabb aabb,
                                float2 & distance)
{
	//AABB is considered as 3 pairs of 2 planes( {x_min, x_max}, {y_min, y_max}, {z_min, z_max} ).
	//t_min is the point of intersection with the closer plane, t_max is the point of intersection with the farther plane.
	//
	//if (rayNormalizedDirection.x < 0.0f), then max.x will be the near plane 
	//and min.x will be the far plane; otherwise, it is reversed.
	//
	//In order for there to be a collision, the t_min and t_max of each pair must overlap.
	//This can be tested for by selecting the highest t_min and lowest t_max and comparing them.
	
	int3 isNegative = isless( rayNormalizedDirection, make_float3(0.0f, 0.0f, 0.0f) );	//isless(x,y) returns (x < y)
	
	//When using vector types, the select() function checks the most signficant bit, 
	//but isless() sets the least significant bit.
	//isNegative <<= 31;

	//select(b, a, condition) == condition ? a : b
	//When using select() with vector types, (condition[i]) is true if its most significant bit is 1
	float3 t_min = float3_difference( select(aabb.high, aabb.low, isNegative), rayOrigin );
	divide_float3(t_min, rayNormalizedDirection);
	float3 t_max = float3_difference( select(aabb.low, aabb.high, isNegative), rayOrigin );
	divide_float3(t_max, rayNormalizedDirection);
	
	float t_min_final = 0.0f;
	float t_max_final = rayLength;

	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	t_min_final = max( t_min.z, max(t_min.y, max(t_min.x, t_min_final)) );
	t_max_final = min( t_max.z, min(t_max.y, min(t_max.x, t_max_final)) );

	distance.x = t_min_final; 
	distance.y = t_max_final;

	return (t_min_final <= t_max_final);
}

__global__ void rayTraverseIterative_kernel(RayInfo * rays,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,								
								float * o_ntests,
								uint numRays)
{
	uint rayIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(rayIndex >= numRays) return;
	
	float3 rayFrom = rays[rayIndex].origin;
	float3 rayTo = rays[rayIndex].destiny;
	float3 rayNormalizedDirection = float3_difference(rayTo, rayFrom);
	
	float rayLength = float3_length(rayNormalizedDirection);
	scale_float3_by(rayNormalizedDirection, 1.0 / rayLength);

	uint stack[B3_PLVBH_TRAVERSE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
		
	float minHitDistanct = 1.f;
	float ntests = 0;
	o_ntests[rayIndex] = ntests;
	float2 testDistance;
	int isFirst = 1;
	int it=0;
	do
	{
		uint internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		
					
		
		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];

		if( rayIntersectsAabb(rayFrom, rayLength, rayNormalizedDirection, bvhNodeAabb, testDistance)  )
		{    
		    
		    //if(testDistance.x < minHitDistanct) 
			
		    
		    
			if(isLeaf)
			{
			    
			    o_ntests[rayIndex] = bvhRigidIndex;
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
			

				if(stackSize + 2 > B3_PLVBH_TRAVERSE_MAX_STACK_SIZE)
				{
					//Error
					break;
				}
				else
				{
				    stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
					stackSize++;
					stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
					stackSize++;
				}
			}

			it++;
			//if(it > 99) break;
		}
		
	}while(stackSize > 0);
	
	// if(minHitDistanct < HUGE_VALUE) {
	    scale_float3_by(rayNormalizedDirection, minHitDistanct);
	    rays[rayIndex].destiny = float3_add(rays[rayIndex].origin, rayNormalizedDirection);
	// }
	
}

__global__ void testRay_kernel(RayInfo * o_ray, Aabb box, float3 pfrom, float h, uint width, uint n)
{
    uint rayIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(rayIndex >= n) return;
	
	o_ray[rayIndex].origin = pfrom;
	
	uint v = rayIndex / width;
	uint u = rayIndex - v * width;
	
	float px = pfrom.x * 0.099f + h * u - h * width * 0.5f;
	float py = - 2.f;
	float pz = pfrom.z * 0.099f + h * v - h * width * 0.5f;
	
	float3 d = make_float3(px - pfrom.x, py - pfrom.y, pz - pfrom.z);
	d = float3_normalize(d);
	
	float l = 100.f;
	/*
	float2 testDistance;
	int stackSize = 1;
	int it = 0;
	while(stackSize) {
	    stackSize--;
	    if( rayIntersectsAabb(pfrom, 1000.f, d, box, testDistance) )
	    {	
	        l = testDistance.x;
	        stackSize++;
	        stackSize++;
		}
		else
		    l = 1.f;
	    
	    it++;
	    if(it>30) break;
	}*/
	
	o_ray[rayIndex].destiny = make_float3(pfrom.x + d.x * l, pfrom.y + d.y * l, pfrom.z + d.z * l);
}

extern "C" void bvhRayTraverseIterative(RayInfo * rays,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,								
								float * o_ntests,
								uint numRays)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numRays, 512);
    
    dim3 grid(nblk, 1, 1);
    
    rayTraverseIterative_kernel<<< grid, block >>>(rays,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,								
								o_ntests,
								numRays);
}

extern "C" void bvhTestRay(RayInfo * o_rays, float3 origin, float boxSize, uint n)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(n * n, 512);
    
    dim3 grid(nblk, 1, 1);
    
    float h = 3.f;
    
    Aabb box;
    box.low.x = -boxSize + 1; box.low.y = 0.f; box.low.z = -boxSize;
    box.high.x = boxSize + 1; box.high.y = boxSize; box.high.z = boxSize;
    testRay_kernel<<< grid, block >>>(o_rays, box, origin, h, n, n * n);
}
