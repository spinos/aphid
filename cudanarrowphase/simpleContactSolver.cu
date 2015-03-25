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

// pointing inside A
inline __device__ float3 normalOnA(const ContactData & contact)
{
    return float3_normalize(float3_from_float4(contact.separateAxis));
}

inline __device__ float computeRelativeVelocity(float3 nA,
                            float3 nB,
                            float3 linearVelocityA, 
                            float3 linearVelocityB,
                            float3 torqueA,
                            float3 torqueB,
                            float3 angularVelocityA, 
                            float3 angularVelocityB)
{
    return float3_dot(linearVelocityA, nA) +
            float3_dot(linearVelocityB, nB) +
            float3_dot(torqueA, angularVelocityA) +
            float3_dot(torqueB, angularVelocityB);
}

inline __device__ float computeDeltaLambda(float & accumulated, float lambda)
{
    const float last = accumulated;
    accumulated += lambda;
    if(accumulated < 0.f) accumulated = 0.f;
	return accumulated - last;
}

inline __device__ void applyImpulse(float3 & dst, float J, float3 N)
{
    dst = float3_add(dst, scale_float3_by(N, J));
}

inline __device__ float computeMassTensor(float3 nA, float3 nB, 
                                        float3 rA, float3 rB,
                                        float invMassA, float invMassB)
{
    float3 torqueA = float3_cross(rA, nA);
    float3 torqueB = float3_cross(rB, nB);
    float3 jmjA = float3_cross( scale_float3_by(torqueA, invMassA), rA );
    float3 jmjB = float3_cross( scale_float3_by(torqueB, invMassB), rB );
    
    return -1.f/(invMassA + invMassB + 
        float3_dot(jmjA, nA) + 
        float3_dot(jmjB, nB));
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

__global__ void setContactConstraint_kernel(float3 * projLinVel,
                                        float3 * projAngVel,
                                        float * lambda,
                                        float * Minv,
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * srcPos,
                                        float3 * srcVel,
                                        uint4 * indices,
                                        uint * pointStarts,
                                        uint * indexStarts,
                                        float * splitMass,
                                        ContactData * contacts,
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint ilft = ind * 2;
	
	lambda[ind] = 0.f;
	
	const uint2 dstInd = splits[ind];
	
	computeBodyVelocities1(pointStarts, indexStarts, indices, pairs[ind].x, srcPos, srcVel, 
	    projLinVel[ilft], projAngVel[ilft]);
	
	computeBodyVelocities1(pointStarts, indexStarts, indices, pairs[ind].y, srcPos, srcVel, 
	    projLinVel[ilft+1], projAngVel[ilft+1]);
	
	ContactData contact = contacts[ind];
	float3 nA = normalOnA(contact);
	float3 nB = float3_reverse(nA);
	
	Minv[ind] = computeMassTensor(nA, nB, contact.localA, contact.localB,
	                            splitMass[dstInd.x], splitMass[dstInd.y]);
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

__global__ void solveContact_kernel(float * lambda,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
                        float3 * linearVelocity,
	                    float3 * angularVelocity,
	                    uint2 * splits,
	                    float * splitMass,
	                    float * Minv,
                        ContactData * contacts,
                        uint maxInd,
                        float * deltaJ,
                        int it)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint2 dstInd = splits[ind];
	
	const uint ilft = ind * 2;
	
	float3 linA = float3_add(linearVelocity[ilft], deltaLinearVelocity[dstInd.x]);
	float3 linB = float3_add(linearVelocity[ilft+1], deltaLinearVelocity[dstInd.y]);
	
	float3 angA = float3_add(angularVelocity[ilft], deltaAngularVelocity[dstInd.x]);
	float3 angB = float3_add(angularVelocity[ilft+1], deltaAngularVelocity[dstInd.y]);
	
	ContactData contact = contacts[ind];

	float3 nA = normalOnA(contact);
	float3 nB = float3_reverse(nA);
// N pointing inside object
// T = r X N	
	float3 torqueA = float3_cross(contact.localA, nA);
	float3 torqueB = float3_cross(contact.localB, nB);
	
	float J = computeRelativeVelocity(nA, nB,
	                        linA, linB,
	                        torqueA, torqueB,
	                        angA, angB);
	
	J *= Minv[ind];
	
	const float invMassA = splitMass[dstInd.x];
	const float invMassB = splitMass[dstInd.y];
	
	float dJ = computeDeltaLambda(lambda[ind], J);
	
	applyImpulse(deltaLinearVelocity[dstInd.x], dJ * invMassA, nA);
	applyImpulse(deltaLinearVelocity[dstInd.y], dJ * invMassB, nB);
	
	applyImpulse(deltaAngularVelocity[dstInd.x], dJ * invMassA, torqueA);
	applyImpulse(deltaAngularVelocity[dstInd.y], dJ * invMassB, torqueB);
	
	deltaJ[ind * JACOBI_NUM_ITERATIONS + it] = dJ;
}

__global__ void averageVelocities_kernel(float3 * linearVelocity,
                        float3 * angularVelocity,
                        uint * bodyCount, 
                        KeyValuePair * srcInd,
                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint c = bodyCount[ind];
	if(c < 1) return;
	
	uint a = srcInd[ind].key;
	
	float3 linSum = linearVelocity[ind];
	float3 angSum = angularVelocity[ind];

	unsigned cur = ind;
// add up backward
	for(;;) {
	    if(cur == maxInd - 1) break;
	    cur++;
	    if(srcInd[cur].key != a) break;
	    
	    linSum = float3_add(linSum, linearVelocity[cur]);
	    angSum = float3_add(angSum, angularVelocity[cur]);
	}

	if(c > 1) {
	    linSum = scale_float3_by(linSum, 1.f / (float)c);
	    angSum = scale_float3_by(angSum, 1.f / (float)c);
	}
	
	linearVelocity[ind] = linSum;
	angularVelocity[ind] = angSum;
	
	cur = ind;
// write backward
	for(;;) {
	    if(cur == maxInd - 1) break;
	    cur++;
	    if(srcInd[cur].key != a) break;
	    
	    linearVelocity[cur] = linSum;
	    angularVelocity[cur] = angSum;
	}
}

__global__ void writePointTetHash_kernel(KeyValuePair * pntTetHash,
	                uint2 * pairs,
	                uint2 * splits,
	                uint * bodyCount,
	                uint4 * indices,
	                uint * pointStart,
                    uint * indexStart,
                    uint numBodies,
	                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint istart = ind * 4;
	
	if(ind >= numBodies) {
	    pntTetHash[istart].key = 1<<30;
        pntTetHash[istart].value = 1<<30;
	    pntTetHash[istart + 1].key = 1<<30;
        pntTetHash[istart + 1].value = 1<<30;
        pntTetHash[istart + 2].key = 1<<30;
        pntTetHash[istart + 2].value = 1<<30;
        pntTetHash[istart + 3].key = 1<<30;
        pntTetHash[istart + 3].value = 1<<30;
        return;
	}
	   
	const unsigned iContact = ind>>1;
	
	uint splitInd = splits[iContact].x;
	uint iBody = pairs[iContact].x;
	
	if(ind & 1) {
	    splitInd = splits[iContact].y;
	    iBody = pairs[iContact].y;
	}
	    
	uint c = bodyCount[splitInd];
	if(c < 1) {
// redundant
        pntTetHash[istart].key = 1<<30;
        pntTetHash[istart].value = 1<<30;
	    pntTetHash[istart + 1].key = 1<<30;
        pntTetHash[istart + 1].value = 1<<30;
        pntTetHash[istart + 2].key = 1<<30;
        pntTetHash[istart + 2].value = 1<<30;
        pntTetHash[istart + 3].key = 1<<30;
        pntTetHash[istart + 3].value = 1<<30;
	}
	else {
	    const uint4 ia = computePointIndex(pointStart, indexStart, indices, iBody);
	    
	    pntTetHash[istart  ].key = ia.x;
	    pntTetHash[istart  ].value = splitInd;
	    pntTetHash[istart+1].key = ia.y;
	    pntTetHash[istart+1].value = splitInd;
	    pntTetHash[istart+2].key = ia.z;
	    pntTetHash[istart+2].value = splitInd;
	    pntTetHash[istart+3].key = ia.w;
	    pntTetHash[istart+3].value = splitInd;
	}
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

void simpleContactSolverSetContactConstraint(float3 * projLinVel,
                                        float3 * projAngVel,
                                        float * lambda,
                                        float * Minv,
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        uint4 * ind,
                                        uint * perObjPointStart,
                                        uint * perObjectIndexStart,
                                        float * splitMass,
                                        ContactData * contacts,
                                        uint numContacts)
{
    uint tpb = CudaBase::LimitNThreadPerBlock(32, 56);

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numContacts, tpb);
    dim3 grid(nblk, 1, 1);
    
    setContactConstraint_kernel<<< grid, block >>>(projLinVel,
                                        projAngVel,
                                        lambda,
                                        Minv,
                                        splits,
                                        pairs,
                                        pos,
                                        vel,
                                        ind,
                                        perObjPointStart,
                                        perObjectIndexStart,
                                        splitMass,
                                        contacts,
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

void simpleContactSolverSolveContact(float * lambda,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
                        float3 * linearVelocity,
	                    float3 * angularVelocity,
	                    uint2 * splits,
	                    float * splitMass,
	                    float * Minv,
                        ContactData * contacts,
                        uint numContacts,
                        float * deltaJ,
                        int it)
{
    uint tpb = CudaBase::LimitNThreadPerBlock(24, 40);

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numContacts, tpb);
    dim3 grid(nblk, 1, 1);
    
    solveContact_kernel<<< grid, block >>>(lambda,
                        deltaLinearVelocity,
                        deltaAngularVelocity,
                        linearVelocity,
	                    angularVelocity,
	                    splits,
	                    splitMass,
	                    Minv,
                        contacts,
                        numContacts,
                        deltaJ,
                        it);
}

void simpleContactSolverAverageVelocities(float3 * linearVelocity,
                        float3 * angularVelocity,
                        uint * bodyCount, 
                        KeyValuePair * srcInd,
                        uint numBodies)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numBodies, 512);
    dim3 grid(nblk, 1, 1);
    
    averageVelocities_kernel<<< grid, block >>>(linearVelocity,
                        angularVelocity,
                        bodyCount, 
                        srcInd,
                        numBodies);
}

void simpleContactSolverWritePointTetHash(KeyValuePair * pntTetHash,
	                uint2 * pairs,
	                uint2 * splits,
	                uint * bodyCount,
	                uint4 * ind,
	                uint * perObjPointStart,
                    uint * perObjectIndexStart,
	                uint numBodies,
	                uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numBodies, 512);
    dim3 grid(nblk, 1, 1);
    
    writePointTetHash_kernel<<< grid, block >>>(pntTetHash,
	                pairs,
	                splits,
	                bodyCount,
	                ind,
	                perObjPointStart,
	                perObjectIndexStart,
	                numBodies,
	                bufLength);
}

}
