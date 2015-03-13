#include "simpleContactSolver_implement.h"
#include <bvh_math.cu>
#include <CudaBase.h>

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

inline __device__ void computeBodyAngularVelocity(float3 & angularVel,
                                                  float3 averageLinearVel,
                                                  float3 * position,
                                                  float3 * velocity,
                                                  uint4 ind)
{
    float3 center;
	float3_average4(center, position, ind);
	
	float3 omega[4];
// omega = r cross v
// v = omega cross r
    omega[0] = float3_cross(float3_difference(position[ind.x], center), float3_difference(velocity[ind.x], averageLinearVel));
    omega[1] = float3_cross(float3_difference(position[ind.y], center), float3_difference(velocity[ind.y], averageLinearVel));
    omega[2] = float3_cross(float3_difference(position[ind.z], center), float3_difference(velocity[ind.z], averageLinearVel));
    omega[3] = float3_cross(float3_difference(position[ind.w], center), float3_difference(velocity[ind.w], averageLinearVel));
    
	float3_average4_direct(angularVel, omega);
}

inline __device__ void computeBodyVelocities1(uint * pointStarts, 
                                                uint * indexStarts, 
                                                uint4 * indices, 
                                                uint ind,
                                                float3 * position,
                                                float3 * velocity, 
                                                float3 & linearVelocity, 
                                                float3 & angularVelocity)
{
    const uint4 ia = computePointIndex(pointStarts, indexStarts, indices, ind);
	
	float3_average4(linearVelocity, velocity, ia);

	computeBodyAngularVelocity(angularVelocity, linearVelocity, position, velocity, ia);
}

inline __device__ void computeBodyVelocities(uint * pointStarts, 
                                                uint * indexStarts, 
                                                uint4 * indices, 
                                                uint2 pair,
                                                float3 * position,
                                                float3 * velocity, 
                                                float3 & linearVelocityA, 
                                                float3 & linearVelocityB,
                                                float3 & angularVelocityA, 
                                                float3 & angularVelocityB)
{
    const uint4 ia = computePointIndex(pointStarts, indexStarts, indices, pair.x);
	const uint4 ib = computePointIndex(pointStarts, indexStarts, indices, pair.y);
	
	float3_average4(linearVelocityA, velocity, ia);
	float3_average4(linearVelocityB, velocity, ib);
	
	float3 centerA;
	float3_average4(centerA, position, ia);
	
	float3 centerB;
	float3_average4(centerB, position, ib);
	
	
}

inline __device__ uint getBodyCountAt(uint ind, uint * count)
{
    uint cur = ind;
    for(;;) {
        if(count[ind] > 0) return count[ind];
        cur--;
    }
}

inline __device__ void 	collide(float3 linearVelocityA, 
                            float3 linearVelocityB,
                            float massA, 
                            float massB,
                            float3 & deltaLinVelA,
                            float3 & deltaLinVelB)
{
    
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
	    dstInd[ind].key = 1<<30;
	    dstInd[ind].value = 1<<30;
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

__global__ void countBody_kernel(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
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
	    if(cur == maxInd - 1) return;
	    cur++;
	    if(srcInd[cur].key != a) return;
	    dstCount[ind]++;
	}	
}

__global__ void computeSplitInvMass_kernel(float * invMass, 
                                        uint * bodyCount, 
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint n = getBodyCountAt(ind, bodyCount);
	
	invMass[ind] = 1.f * (float)n;
}

__global__ void setContactConstraint_kernel(float3 * linVelA,
                                             float3 * linVelB,
                                        float3 * angVelA,
                                        float3 * angVelB,
                                        float * lambda,
                                        float * invMass, 
                                        float * splitInvMass, 
                                        uint2 * splits, 
                                        uint2 * pairs,
                                        float3 * srcPos,
                                        float3 * srcVel,
                                        uint4 * indices,
                                        uint * pointStarts,
                                        uint * indexStarts,
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint2 massInd = splits[ind];
	
	float invMassA = splitInvMass[massInd.x];
	float invMassB = splitInvMass[massInd.y];
	
	invMass[ind] = 1.f / (invMassA + invMassB);
	lambda[ind] = 0.f;
	
	computeBodyVelocities1(pointStarts, indexStarts, indices, pairs[ind].x, srcPos, srcVel, 
	    linVelA[ind], angVelA[ind]);
	
	computeBodyVelocities1(pointStarts, indexStarts, indices, pairs[ind].y, srcPos, srcVel, 
	    linVelB[ind], angVelB[ind]);
}

__global__ void clearDeltaVelocity_kernel(float3 * deltaLinVel, 
                                        float3 * deltaAngVel, 
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	deltaLinVel[ind] = make_float3(0.f, 0.f, 0.f);
	deltaAngVel[ind] = make_float3(0.f, 0.f, 0.f);
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

__global__ void solveContact_kernel(float3 * deltaLinVel,
                        float3 * deltaAngVel,
	                    uint2 * splits,
	                    float * splitMass,
	                    float3 * srcVelocity,
                    uint2 * pairs,
                    uint4 * indices,
                    uint * pointStarts,
                    uint * indexStarts,
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3 linearVelocityA, linearVelocityB, angularVelocityA, angularVelocityB;
	computeBodyVelocities(pointStarts, indexStarts, indices, pairs[ind], srcVelocity, srcVelocity,
	    linearVelocityA, linearVelocityB, angularVelocityA, angularVelocityB);
	
	const uint2 dstInd = splits[ind];
	float massA = splitMass[dstInd.x];
	float massB = splitMass[dstInd.y];
	
	collide(linearVelocityA, linearVelocityB,
	        massA, massB,
	        deltaLinVel[dstInd.x],
	        deltaLinVel[dstInd.y]
	        );
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

void simpleContactSolverCountBody(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint num)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(num, 512);
    dim3 grid(nblk, 1, 1);
    
    countBody_kernel<<< grid, block >>>(dstCount,
                                     srcInd, 
                                       num);
}

void simpleContactSolverComputeSplitInverseMass(float * invMass, 
                                        uint * bodyCount, 
                                        uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    
    computeSplitInvMass_kernel<<< grid, block >>>(invMass,
                                     bodyCount, 
                                       bufLength);
}

void simpleContactSolverSetContactConstraint(float3 * linVelA,
                                             float3 * linVelB,
                                        float3 * angVelA,
                                        float3 * angVelB,
                                        float * lambda,
                                        float * invMass, 
                                        float * splitInvMass, 
                                        uint2 * splits, 
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        uint4 * ind,
                                        uint * perObjPointStart,
                                        uint * perObjectIndexStart,
                                        uint numContacts)
{
    uint tpb = CudaBase::LimitNThreadPerBlock(30, 60);
    
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numContacts, tpb);
    dim3 grid(nblk, 1, 1);
    
    setContactConstraint_kernel<<< grid, block >>>(linVelA,
                                        linVelB,
                                        angVelA,
                                        angVelB,
                                        lambda,
                                        invMass, 
                                        splitInvMass, 
                                        splits, 
                                        pairs,
                                        pos,
                                        vel,
                                        ind,
                                        perObjPointStart,
                                        perObjectIndexStart,
                                        numContacts);
}

void simpleContactSolverClearDeltaVelocity(float3 * deltaLinVel, 
                                        float3 * deltaAngVel, 
                                        uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    
    clearDeltaVelocity_kernel<<< grid, block >>>(deltaLinVel,
                                     deltaAngVel, 
                                       bufLength);
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

void simpleContactSolverSolveContact(float3 * deltaLinVel,
	                    float3 * deltaAngVel,
	                    uint2 * splits,
	                    float * splitMass,
	                    float3 * srcVelocity,
                    uint2 * pairs,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numContacts)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numContacts, 512);
    dim3 grid(nblk, 1, 1);
    
    solveContact_kernel<<< grid, block >>>(deltaLinVel,
	                    deltaAngVel,
	                    splits,
	                    splitMass,
	                    srcVelocity,
                        pairs,
                        indices,
                        objectPointStarts,
                        objectIndexStarts,
                        numContacts);
}

}
