#include "traverseBvh_implement.h"
#include <radixsort_implement.h>
#include <bvh_math.cu>
#include <CudaBase.h>

#define B3_PLVBH_TRAVERSE_MAX_STACK_SIZE 128

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
