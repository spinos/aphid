#include "createBvh_implement.h"
#include "bvh_math.cuh"
#include <CudaBase.h>
//Set so that it is always greater than the actual common prefixes, and never selected as a parent node.
//If there are no duplicates, then the highest common prefix is 32 or 64, depending on the number of bits used for the z-curve.
//Duplicate common prefixes increase the highest common prefix at most by the number of bits used to index the leaf node.
//Since 32 bit ints are used to index leaf nodes, the max prefix is 64(32 + 32 bit z-curve) or 96(32 + 64 bit z-curve).
#define B3_PLBVH_INVALID_COMMON_PREFIX 128
#define B3_PLBVH_ROOT_NODE_MARKER -1
#define CALC_TETRA_AABB_NUM_THREADS 512

inline __device__ int computeCommonPrefixLength(uint64 i, uint64 j) 
{ return (int)__clzll(i ^ j); }

inline __device__ uint64 computeCommonPrefix(uint64 i, uint64 j) 
{
	//This function only needs to return (i & j) in order for the algorithm to work,
	//but it may help with debugging to mask out the lower bits.

	uint64 commonPrefixLength = (uint64)computeCommonPrefixLength(i, j);

	uint64 sharedBits = i & j;
	
	//Set all bits after the common prefix to 0
	uint64 bitmask = ((uint64)(~0)) << (64 - commonPrefixLength);	
	
	return sharedBits & bitmask;
}

//Same as computeCommonPrefixLength(), but allows for prefixes with different lengths
inline __device__ int getSharedPrefixLength(uint64 prefixA, int prefixLengthA, uint64 prefixB, int prefixLengthB)
{
    return min( computeCommonPrefixLength(prefixA, prefixB), min(prefixLengthA, prefixLengthB) );
}

inline __device__ uint64 upsample(uint a, uint b) 
{ return ((uint64)a << 32) | (uint64)b; }

// Expands a 10-bit integer into 30 bits 
// by inserting 2 zeros after each bit. 
inline __device__ uint expandBits(uint v) 
{ 
    v = (v * 0x00010001u) & 0xFF0000FFu; 
    v = (v * 0x00000101u) & 0x0F00F00Fu; 
    v = (v * 0x00000011u) & 0xC30C30C3u; 
    v = (v * 0x00000005u) & 0x49249249u; 
    return v; 
}

// Calculates a 30-bit Morton code for the 
// given 3D point located within the unit cube [0,1].
inline __device__ uint morton3D(float x, float y, float z) 
{ 
    x = min(max(x * 1024.0f, 0.0f), 1023.0f); 
    y = min(max(y * 1024.0f, 0.0f), 1023.0f); 
    z = min(max(z * 1024.0f, 0.0f), 1023.0f); 
    uint xx = expandBits((uint)x); 
    uint yy = expandBits((uint)y); 
    uint zz = expandBits((uint)z); 
    return (xx <<2) + (yy <<1) + zz; 
} 

inline __device__ void normalizeByBoundary(float & x, float low, float width)
{
	if(x < low) x = 0.0f;
	else if(x > low + width) x = 1.0f;
	else {
		if(width < TINY_VALUE2) x = 0.0f;
		else x = (x - low) / width;
	}
}

__global__ void calculateAabbsTetrahedron2_kernel(Aabb *dst, float3 * pos, float3 * vel, float h, 
                                                            uint4 * tetrahedronVertices, 
                                                            unsigned maxNumPerTetVs)
{
    __shared__ float3 sP0[CALC_TETRA_AABB_NUM_THREADS];
    __shared__ float3 sP1[CALC_TETRA_AABB_NUM_THREADS];
    
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxNumPerTetVs) return;
	
	uint itet = idx>>2;
	uint ivert = idx & 3;
	uint * vtet = & tetrahedronVertices[itet].x;
	
	uint iv = vtet[ivert];
	
	sP0[threadIdx.x] = pos[iv];
	sP1[threadIdx.x] = float3_progress(pos[iv], vel[iv], h);
	__syncthreads();
	
	if(ivert > 0) return;
	
	Aabb res;
	resetAabb(res);
	
	expandAabb(res, sP0[threadIdx.x]);
	expandAabb(res, sP1[threadIdx.x]);
	expandAabb(res, sP0[threadIdx.x + 1]);
	expandAabb(res, sP1[threadIdx.x + 1]);
	expandAabb(res, sP0[threadIdx.x + 2]);
	expandAabb(res, sP1[threadIdx.x + 2]);
	expandAabb(res, sP0[threadIdx.x + 3]);
	expandAabb(res, sP1[threadIdx.x + 3]);
	
	dst[itet] = res;
}

__global__ void calculateAabbsTetrahedron_kernel(Aabb *dst, float3 * cvs, uint4 * tets, unsigned maxTetInd)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxTetInd) return;
	
	uint4 t = tets[idx];

	Aabb res;
	resetAabb(res);
	expandAabb(res, cvs[t.x]);
	expandAabb(res, cvs[t.y]);
	expandAabb(res, cvs[t.z]);
	expandAabb(res, cvs[t.w]);
	
	dst[idx] = res;
}

__global__ void calculateAabbsTriangle_kernel(Aabb *dst, float3 * cvs, uint3 * tris, unsigned maxTriInd)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxTriInd) return;
	
	uint3 t = tris[idx];

	Aabb res;
	resetAabb(res);
	expandAabb(res, cvs[t.x]);
	expandAabb(res, cvs[t.y]);
	expandAabb(res, cvs[t.z]);
	
	dst[idx] = res;
}

__global__ void calculateAabbs_kernel(Aabb *dst, float3 * cvs, EdgeContact * edges, unsigned maxEdgeInd, unsigned maxVertInd)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxEdgeInd) return;
	
	EdgeContact e = edges[idx];
	unsigned v0 = e.v[0];
	unsigned v1 = e.v[1];
	unsigned v2 = e.v[2];
	unsigned v3 = e.v[3];
	
	Aabb res;
	resetAabb(res);
	if(v0 < maxVertInd) expandAabb(res, cvs[v0]);
	if(v1 < maxVertInd) expandAabb(res, cvs[v1]);
	if(v2 < maxVertInd) expandAabb(res, cvs[v2]);
	if(v3 < maxVertInd) expandAabb(res, cvs[v3]);
	
	dst[idx] = res;
}

__global__ void resetLeafHash_kernel(KeyValuePair *dst, uint bufSize)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx >= bufSize) return;
	
	dst[idx].key = 1073741825; // 2^30 + 1
	dst[idx].value = idx;
}

__global__ void calculateLeafHash_kernel(KeyValuePair *dst, Aabb * leafBoxes, uint maxInd, uint bufSize, 
		Aabb * boundary)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx >= bufSize) return;
	
	dst[idx].value = idx;
	if(idx >= maxInd) {
		dst[idx].key = 1073741823; // 2^30 - 1
		return;
	}
	
	float3 c = centroidOfAabb(leafBoxes[idx]);
	
	float side = longestSideOfAabb(*boundary);
	float3 d = float3_difference(boundary->high, boundary->low);
	if(d.x < side) d.x = side;
	if(d.y < side) d.y = side;
	if(d.z < side) d.z = side;
	normalizeByBoundary(c.x, boundary->low.x, d.x);
	normalizeByBoundary(c.y, boundary->low.y, d.y);
	normalizeByBoundary(c.z, boundary->low.z, d.z);
	
	dst[idx].key = morton3D(c.x, c.y, c.z);
}

__global__ void computeAdjacentPairCommonPrefix_kernel(KeyValuePair * mortonCodesAndAabbIndices,
											uint64* out_commonPrefixes,
											int* out_commonPrefixLengths,
											uint numInternalNodes)
{
	uint internalNodeIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(internalNodeIndex >= numInternalNodes) return;
	
	//Here, (internalNodeIndex + 1) is never out of bounds since it is a leaf node index,
	//and the number of internal nodes is always numLeafNodes - 1
	uint leftLeafIndex = internalNodeIndex;
	uint rightLeafIndex = internalNodeIndex + 1;
	
	uint leftLeafMortonCode = mortonCodesAndAabbIndices[leftLeafIndex].key;
	uint rightLeafMortonCode = mortonCodesAndAabbIndices[rightLeafIndex].key;
	
	//Binary radix tree construction algorithm does not work if there are duplicate morton codes.
	//Append the index of each leaf node to each morton code so that there are no duplicates.
	//The algorithm also requires that the morton codes are sorted in ascending order; this requirement
	//is also satisfied with this method, as (leftLeafIndex < rightLeafIndex) is always true.
	//
	//
	uint64 nonduplicateLeftMortonCode = upsample(leftLeafMortonCode, leftLeafIndex);
	uint64 nonduplicateRightMortonCode = upsample(rightLeafMortonCode, rightLeafIndex);
	
	out_commonPrefixes[internalNodeIndex] = computeCommonPrefix(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
	out_commonPrefixLengths[internalNodeIndex] = computeCommonPrefixLength(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
}

__global__ void connectLeafNodesToInternalTree_kernel(int* commonPrefixLengths, int* out_leafNodeParentNodes,
											int2* out_childNodes, int numLeafNodes)
{
	int leafNodeIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if (leafNodeIndex >= numLeafNodes) return;
	
	int numInternalNodes = numLeafNodes - 1;
	
	int leftSplitIndex = leafNodeIndex - 1;
	int rightSplitIndex = leafNodeIndex;
	
	int leftCommonPrefix = (leftSplitIndex >= 0) ? commonPrefixLengths[leftSplitIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
	int rightCommonPrefix = (rightSplitIndex < numInternalNodes) ? commonPrefixLengths[rightSplitIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
	
	//Parent node is the highest adjacent common prefix that is lower than the node's common prefix
	//Leaf nodes are considered as having the highest common prefix
	int isLeftHigherCommonPrefix = (leftCommonPrefix > rightCommonPrefix);
	
	//Handle cases for the edge nodes; the first and last node
	//For leaf nodes, leftCommonPrefix and rightCommonPrefix should never both be B3_PLBVH_INVALID_COMMON_PREFIX
	if(leftCommonPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherCommonPrefix = false;
	if(rightCommonPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherCommonPrefix = true;
	
	int parentNodeIndex = (isLeftHigherCommonPrefix) ? leftSplitIndex : rightSplitIndex;
	out_leafNodeParentNodes[leafNodeIndex] = parentNodeIndex;
	
	int isRightChild = (isLeftHigherCommonPrefix);	//If the left node is the parent, then this node is its right child and vice versa
	
	//out_childNodesAsInt[0] == int2.x == left child
	//out_childNodesAsInt[1] == int2.y == right child
	int isLeaf = 1;
	int* out_childNodesAsInt = (int*)(&out_childNodes[parentNodeIndex]);
	out_childNodesAsInt[isRightChild] = getIndexWithInternalNodeMarkerSet(isLeaf, leafNodeIndex);	
}

__global__ void connectInternalTreeNodes_kernel(uint64* commonPrefixes, int* commonPrefixLengths,
												int2* out_childNodes,
												int* out_internalNodeParentNodes,
												int* out_rootNodeIndex,
												uint numInternalNodes)
{
	uint internalNodeIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(internalNodeIndex >= numInternalNodes) return;
	
	uint64 nodePrefix = commonPrefixes[internalNodeIndex];
	int nodePrefixLength = commonPrefixLengths[internalNodeIndex];
/*	
	int leftIndex = -1;
	int rightIndex = -1;
	
	//Find nearest element to left with a lower common prefix
	for(int i = internalNodeIndex - 1; i >= 0; --i)
	{
		int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, commonPrefixes[i], commonPrefixLengths[i]);
		if(nodeLeftSharedPrefixLength < nodePrefixLength)
		{
			leftIndex = i;
			break;
		}
	}
	
	//Find nearest element to right with a lower common prefix
	for(int i = internalNodeIndex + 1; i < numInternalNodes; ++i)
	{
		int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, commonPrefixes[i], commonPrefixLengths[i]);
		if(nodeRightSharedPrefixLength < nodePrefixLength)
		{
			rightIndex = i;
			break;
		}
	}
*/

	// binary search

	//Find nearest element to left with a lower common prefix
	int leftIndex = -1;
	{
		int lower = 0;
		int upper = internalNodeIndex - 1;
		
		while(lower <= upper)
		{
			int mid = (lower + upper) / 2;
			uint64 midPrefix = commonPrefixes[mid];
			int midPrefixLength = commonPrefixLengths[mid];
			
			int nodeMidSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, midPrefix, midPrefixLength);
			if(nodeMidSharedPrefixLength < nodePrefixLength) 
			{
				int right = mid + 1;
				if(right < internalNodeIndex)
				{
					uint64 rightPrefix = commonPrefixes[right];
					int rightPrefixLength = commonPrefixLengths[right];
					
					int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, rightPrefix, rightPrefixLength);
					if(nodeRightSharedPrefixLength < nodePrefixLength) 
					{
						lower = right;
						leftIndex = right;
					}
					else 
					{
						leftIndex = mid;
						break;
					}
				}
				else 
				{
					leftIndex = mid;
					break;
				}
			}
			else upper = mid - 1;
		}
	}
	
	//Find nearest element to right with a lower common prefix
	int rightIndex = -1;
	{
		int lower = internalNodeIndex + 1;
		int upper = numInternalNodes - 1;
		
		while(lower <= upper)
		{
			int mid = (lower + upper) / 2;
			uint64 midPrefix = commonPrefixes[mid];
			int midPrefixLength = commonPrefixLengths[mid];
			
			int nodeMidSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, midPrefix, midPrefixLength);
			if(nodeMidSharedPrefixLength < nodePrefixLength) 
			{
				int left = mid - 1;
				if(left > internalNodeIndex)
				{
					uint64 leftPrefix = commonPrefixes[left];
					int leftPrefixLength = commonPrefixLengths[left];
				
					int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, leftPrefix, leftPrefixLength);
					if(nodeLeftSharedPrefixLength < nodePrefixLength) 
					{
						upper = left;
						rightIndex = left;
					}
					else 
					{
						rightIndex = mid;
						break;
					}
				}
				else 
				{
					rightIndex = mid;
					break;
				}
			}
			else lower = mid + 1;
		}
	}

	//Select parent
	{
		int leftPrefixLength = (leftIndex != -1) ? commonPrefixLengths[leftIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
		int rightPrefixLength =  (rightIndex != -1) ? commonPrefixLengths[rightIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
		
		int isLeftHigherPrefixLength = (leftPrefixLength > rightPrefixLength);
		
		if(leftPrefixLength == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = false;
		else if(rightPrefixLength == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = true;
		
		int parentNodeIndex = (isLeftHigherPrefixLength) ? leftIndex : rightIndex;
		
		int isRootNode = (leftIndex == -1 && rightIndex == -1);
		out_internalNodeParentNodes[internalNodeIndex] = (!isRootNode) ? parentNodeIndex : B3_PLBVH_ROOT_NODE_MARKER;
		
		int isLeaf = 0;
		if(!isRootNode)
		{
			int isRightChild = (isLeftHigherPrefixLength);	//If the left node is the parent, then this node is its right child and vice versa
			
			//out_childNodesAsInt[0] == int2.x == left child
			//out_childNodesAsInt[1] == int2.y == right child
			int* out_childNodesAsInt = (int*)(&out_childNodes[parentNodeIndex]);
			out_childNodesAsInt[isRightChild] = getIndexWithInternalNodeMarkerSet(isLeaf, internalNodeIndex);
		}
		else *out_rootNodeIndex = getIndexWithInternalNodeMarkerSet(isLeaf, internalNodeIndex);
	}
}

__global__ void findDistanceFromRoot_kernel(int* rootNodeIndex, int* internalNodeParentNodes,
									// int* out_maxDistanceFromRoot, 
									int* out_distanceFromRoot, 
									uint numInternalNodes)
{
	uint internalNodeIndex = blockIdx.x*blockDim.x + threadIdx.x;
	
	// need reduce here if( internalNodeIndex == 0 ) atomic_xchg(out_maxDistanceFromRoot, 0);

	if(internalNodeIndex >= numInternalNodes) return;
	
	int distanceFromRoot = 0;
	{
		int parentIndex = internalNodeParentNodes[internalNodeIndex];
		while(parentIndex != B3_PLBVH_ROOT_NODE_MARKER)
		{
			parentIndex = internalNodeParentNodes[parentIndex];
			++distanceFromRoot;
		}
	}
	out_distanceFromRoot[internalNodeIndex] = distanceFromRoot;
	
	/* need reduce here
	__local int localMaxDistanceFromRoot;
	if( get_local_id(0) == 0 ) localMaxDistanceFromRoot = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	atomic_max(&localMaxDistanceFromRoot, distanceFromRoot);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if( get_local_id(0) == 0 ) atomic_max(out_maxDistanceFromRoot, localMaxDistanceFromRoot);
	*/
}

__global__ void formInternalNodeAabbsAtDistance_kernel(int * distanceFromRoot, KeyValuePair * mortonCodesAndAabbIndices,
												int2 * childNodes,
												Aabb * leafNodeAabbs, Aabb * internalNodeAabbs,
												int * maxChildInd,
												int maxDistanceFromRoot, int processedDistance, 
												uint numInternalNodes)
{
	uint internalNodeIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(internalNodeIndex >= numInternalNodes) return;
	
	int distance = distanceFromRoot[internalNodeIndex];
	
	if(distance == processedDistance)
	{
		int leftChildIndex = childNodes[internalNodeIndex].x;
		int rightChildIndex = childNodes[internalNodeIndex].y;
		
		int isLeftChildLeaf = isLeafNode(leftChildIndex);
		int isRightChildLeaf = isLeafNode(rightChildIndex);
		
		leftChildIndex = getIndexWithInternalNodeMarkerRemoved(leftChildIndex);
		rightChildIndex = getIndexWithInternalNodeMarkerRemoved(rightChildIndex);
		
		//leftRigidIndex/rightRigidIndex is not used if internal node
		int leftRigidIndex = (isLeftChildLeaf) ? mortonCodesAndAabbIndices[leftChildIndex].value : -1;
		int rightRigidIndex = (isRightChildLeaf) ? mortonCodesAndAabbIndices[rightChildIndex].value : -1;
		
		Aabb leftChildAabb = (isLeftChildLeaf) ? leafNodeAabbs[leftRigidIndex] : internalNodeAabbs[leftChildIndex];
		Aabb rightChildAabb = (isRightChildLeaf) ? leafNodeAabbs[rightRigidIndex] : internalNodeAabbs[rightChildIndex];
		
		int elmLimit = -1;
		if(isLeftChildLeaf) elmLimit = max(elmLimit, leftRigidIndex);
		else elmLimit = max(elmLimit, maxChildInd[leftChildIndex]);
		
		if(isRightChildLeaf) elmLimit = max(elmLimit, rightRigidIndex);
		else elmLimit = max(elmLimit, maxChildInd[rightChildIndex]);
		
		maxChildInd[internalNodeIndex] = elmLimit;
		
		Aabb mergedAabb = leftChildAabb;
		expandAabb(mergedAabb, rightChildAabb);
		internalNodeAabbs[internalNodeIndex] = mergedAabb;
	}
}

extern "C" void bvhCalculateLeafAabbsTetrahedron2(Aabb *dst, float3 * pos, float3 * vel, float timeStep, uint4 * tets, unsigned numTetrahedrons)
{
    int tpb = CALC_TETRA_AABB_NUM_THREADS;

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numTetrahedrons<<2, tpb);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbsTetrahedron2_kernel<<< grid, block >>>(dst, pos, vel, timeStep, tets, numTetrahedrons<<2);
}

extern "C" void bvhCalculateLeafAabbsTetrahedron(Aabb *dst, float3 * cvs, uint4 * tets, unsigned numTetrahedrons)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numTetrahedrons, 512);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbsTetrahedron_kernel<<< grid, block >>>(dst, cvs, tets, numTetrahedrons);
}

extern "C" void bvhCalculateLeafAabbsTriangle(Aabb *dst, float3 * cvs, uint3 * tris, unsigned numTriangles)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numTriangles, 512);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbsTriangle_kernel<<< grid, block >>>(dst, cvs, tris, numTriangles);
}

extern "C" void bvhCalculateLeafAabbs(Aabb *dst, float3 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numEdges, 512);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbs_kernel<<< grid, block >>>(dst, cvs, edges, numEdges, numVertices);
}

extern "C" void bvhResetLeafHash(KeyValuePair * dst, uint buffSize)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(buffSize, 512);
    dim3 grid(nblk, 1, 1);
	resetLeafHash_kernel<<< grid, block >>>(dst, buffSize);
}

extern "C" void bvhCalculateLeafHash(KeyValuePair * dst, Aabb * leafBoxes, uint numLeaves, uint buffSize, 
			Aabb * bigBox)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(buffSize, 512);
    
    dim3 grid(nblk, 1, 1);
	calculateLeafHash_kernel<<< grid, block >>>(dst, leafBoxes, numLeaves, buffSize, bigBox);
}

extern "C" void bvhComputeAdjacentPairCommonPrefix(KeyValuePair * mortonCode,
													uint64 * o_commonPrefix,
													int * o_commonPrefixLength,
													uint numInternalNodes)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numInternalNodes, 512);
    
    dim3 grid(nblk, 1, 1);
	computeAdjacentPairCommonPrefix_kernel<<< grid, block >>>(mortonCode, o_commonPrefix, o_commonPrefixLength, numInternalNodes);
}

extern "C" void bvhConnectLeafNodesToInternalTree(int * commonPrefixLengths, 
								int * o_leafNodeParentIndex,
								int2 * o_internalNodeChildIndex, 
								uint numLeafNodes)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numLeafNodes, 512);
    
    dim3 grid(nblk, 1, 1);
    connectLeafNodesToInternalTree_kernel<<< grid, block >>>(commonPrefixLengths, o_leafNodeParentIndex, o_internalNodeChildIndex, numLeafNodes);
}

extern "C" void bvhConnectInternalTreeNodes(uint64 * commonPrefix, int * commonPrefixLengths,
											int2 * o_internalNodeChildIndex,
											int * o_internalNodeParentIndex,
											int * o_rootNodeIndex,
											uint numInternalNodes)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numInternalNodes, 512);
    
    dim3 grid(nblk, 1, 1);
    connectInternalTreeNodes_kernel<<< grid, block >>>(commonPrefix, commonPrefixLengths, o_internalNodeChildIndex, o_internalNodeParentIndex, o_rootNodeIndex, numInternalNodes);
}

extern "C" void bvhFindDistanceFromRoot(int* rootNodeIndex, int* internalNodeParentNodes,
									int* out_distanceFromRoot, 
									uint numInternalNodes)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numInternalNodes, 512);
    
    dim3 grid(nblk, 1, 1);
    findDistanceFromRoot_kernel<<< grid, block >>>(rootNodeIndex, internalNodeParentNodes,
								out_distanceFromRoot, 
								numInternalNodes);
}

extern "C" void bvhFormInternalNodeAabbsAtDistance(int * distanceFromRoot, KeyValuePair * mortonCodesAndAabbIndices,
												int2 * childNodes,
												Aabb * leafNodeAabbs, Aabb * internalNodeAabbs,
												int * maxChildInd,
												int maxDistanceFromRoot, int processedDistance, 
												uint numInternalNodes)
{
	int tpb = CudaBase::LimitNThreadPerBlock(18, 52);

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numInternalNodes, tpb);
    
    dim3 grid(nblk, 1, 1);
    formInternalNodeAabbsAtDistance_kernel<<< grid, block >>>(distanceFromRoot, mortonCodesAndAabbIndices,
										childNodes,
										leafNodeAabbs, internalNodeAabbs,
										maxChildInd,
										maxDistanceFromRoot, processedDistance, 
										numInternalNodes);
}

