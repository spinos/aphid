#include "simpleContactSolver_implement.h"
#include <bvh_math.cu>

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

__global__ void writeContactIndex_kernel(KeyValuePair * dstInd, 
                                    uint * srcInd, 
                                    uint n, uint bufferLength)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= bufferLength) return;
	
	if(ind < n) {
	    dstInd[ind].key = srcInd[ind];
	    dstInd[ind].value = ind >> 1;
	}
	else {
	    dstInd[ind].key = 1e30;
	    dstInd[ind].value = 1e30;
	}
}

__global__ void computeSplitBufLoc_kernel(uint2 * splits, 
                                    uint2 * srcPairs, 
                                    KeyValuePair * bodyPairHash, 
                                    uint bufLength)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= bufLength) return;
	
	const uint dstLoc = bodyPairHash[ind].value;
	if(srcPairs[dstLoc].x == bodyPairHash[ind].key) {
	    splits[dstLoc].x = ind;
	}
	else {
	    splits[dstLoc].y = ind;
	}
}

__global__ void countUniqueBody_kernel(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint num, uint bufLength)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= bufLength) return;

	if(ind >= num) {
	    dstCount[ind] = 0;
	    return;
	}
	
	dstCount[ind] = 0;
	
	const uint a = srcInd[ind].key;
	
	int isFirst = 0;
	
	if(ind < 1) isFirst = 1;
	else if(srcInd[ind - 1].key != a) isFirst = 1;
	
	if(!isFirst) return;
	
	dstCount[ind] = 1;

	unsigned cur = ind;
// check backward
	for(;;) {
	    if(cur == num - 1) return;
	    cur++;
	    if(srcInd[cur].key != a) return;
	    dstCount[ind]++;
	}	
}

__global__ void stopAtContact_kernel(float3 * dstVelocity,
                        uint2 * pairs,
                        uint4 * indices,
                        uint * pointStarts,
                        uint * indexStarts,
                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint4 ia = computePointIndex(pointStarts, indexStarts, indices, pairs[ind].x);
	const uint4 ib = computePointIndex(pointStarts, indexStarts, indices, pairs[ind].y);
	
	dstVelocity[ia.x] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ia.y] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ia.z] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ia.w] = make_float3(0.f, 0.f, 0.f);
	
	dstVelocity[ib.x] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ib.y] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ib.z] = make_float3(0.f, 0.f, 0.f);
	dstVelocity[ib.w] = make_float3(0.f, 0.f, 0.f);
}

extern "C" {
    
void simpleContactSolverWriteContactIndex(KeyValuePair * dstInd, 
                                    uint * srcInd, 
                                    uint n, uint bufferLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufferLength, 512);
    dim3 grid(nblk, 1, 1);
    
    writeContactIndex_kernel<<< grid, block >>>(dstInd, 
                                                srcInd,
                                                n, bufferLength);
}

void simpleContactSolverComputeSplitBufLoc(uint2 * splits, 
                                    uint2 * srcPairs, 
                                    KeyValuePair * bodyPairHash, 
                                    uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    
    computeSplitBufLoc_kernel<<< grid, block >>>(splits, 
                                        srcPairs, 
                                        bodyPairHash, 
                                        bufLength);
}

void simpleContactSolverCountUniqueBody(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint num, uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    
    countUniqueBody_kernel<<< grid, block >>>(dstCount,
                                     srcInd, 
                                       num, bufLength);
}

void simpleContactSolverStopAtContact(float3 * dstVelocity,
                        uint2 * pairs,
                        uint4 * indices,
                        uint * objectPointStarts,
                        uint * objectIndexStarts,
                        uint numContacts)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numContacts, 512);
    dim3 grid(nblk, 1, 1);
    
    stopAtContact_kernel<<< grid, block >>>(dstVelocity, 
                        pairs,
                        indices,
                        objectPointStarts,
                        objectIndexStarts,
                        numContacts);
}

}
