#include "simpleContactSolver_implement.h"
#include "bvh_math.cuh"
#include "barycentric.cu"
#include "stripedModel.cu"
#include <CudaBase.h>
#define SETCONSTRAINT_TPB 128
#define SOLVECONTACT_TPB 256
#define DEFORMABILITY 0.0134f
#define ENABLE_DEFORMABILITY 0
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

inline __device__ BarycentricCoordinate localCoordinate(uint4 ia,
                                    float3 * position,
                                    float3 localP)
{
    float3 q;
	float3_average4(q, position, ia);
	q = float3_add(q, localP);
	
	return getBarycentricCoordinate4i(q, position, ia);
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

inline __device__ void computeBodyCenter(float3 & center, 
                                uint iBody,
                                float3 * position,
                                uint4 * indices,
                                uint * pointStarts,
                                uint * indexStarts)
{
    const uint4 iVert = computePointIndex(pointStarts, indexStarts, indices, iBody);
	float3_average4(center, position, iVert);
}

inline __device__ uint getBodyCountAt(uint ind, uint * count)
{
    uint cur = ind;
    for(;;) {
        if(count[cur] > 0) return count[cur];
        cur--;
    }
}

// pointing inside A
inline __device__ float3 normalOnA(const ContactData & contact)
{
    return float3_normalize(float3_from_float4(contact.separateAxis));
}

inline __device__ float computeRelativeVelocity1(float3 nA,
                            float3 nB,
                            float3 linearVelocityA, 
                            float3 linearVelocityB)
{
    return float3_dot(linearVelocityA, nA) +
            float3_dot(linearVelocityB, nB);
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
            float3_dot(linearVelocityB, nB);// +
            // float3_dot(torqueA, angularVelocityA) +
            // float3_dot(torqueB, angularVelocityB);
}

inline __device__ void applyImpulse(float3 & dst, float J, float3 N)
{
    dst = float3_add(dst, scale_float3_by(N, J));
}

inline __device__ float computeMassTensor(float3 nA, float3 nB, 
                                        float3 rA, float3 rB,
                                        float3 torqueA, float3 torqueB,
                                        float invMassA, float invMassB)
{
    float3 jmjA = float3_cross( scale_float3_by(torqueA, invMassA), rA );
    float3 jmjB = float3_cross( scale_float3_by(torqueB, invMassB), rB );
    
    return -1.f/(invMassA + invMassB +  
         invMassA + invMassB);    
         // invMassA * float3_dot(jmjA, nA) + 
         // invMassB * float3_dot(jmjB, nB));
}

inline __device__ void deformMotion(float3 & dst,
                                    float3 r, 
                                    float3 n,
                                    float3 omega)
{
// v = omega X r 
    dst = float3_cross(omega, r);
    float l = float3_length2(dst);
    float lr = float3_length(r);
// limit size of rotation
    if(l> lr * .59f) l = lr * .59f;
    if(l>1e-2) dst = float3_normalize(dst);
    dst = scale_float3_by(dst, l);
    dst = float3_add(dst, scale_float3_by(n, lr));
    dst = scale_float3_by(dst, DEFORMABILITY);
}

inline __device__ void addDeltaVelocity(float3 & dst, 
        float3 deltaLinearVelocity,
        float3 deltaAngularVelocity,
        float3 normal, 
        float3 r,
        BarycentricCoordinate * coord)
{
    dst = float3_add(dst, deltaLinearVelocity);
#if ENABLE_DEFORMABILITY
    float3 vRot;
    deformMotion(vRot, r, normal, deltaAngularVelocity);
    
// distribure by weight to vex, then sum by weight from vex
    float wei = coord->x * coord->x;
    wei = wei > 1.f ? 1.f : wei;
    dst = float3_add(dst, scale_float3_by(vRot, wei));
    
    wei = coord->y * coord->y;
    wei = wei > 1.f ? 1.f : wei;
    dst = float3_add(dst, scale_float3_by(vRot, wei));
    
    wei = coord->z * coord->z;
    wei = wei > 1.f ? 1.f : wei;
    dst = float3_add(dst, scale_float3_by(vRot, wei));
    
    wei = coord->w * coord->w;
    wei = wei > 1.f ? 1.f : wei;
    dst = float3_add(dst, scale_float3_by(vRot, wei));
#endif
}

inline __device__ float getPntTetWeight(uint pnt, 
                            uint4 tet, 
                            BarycentricCoordinate coord)
{
    if(pnt == tet.x) return coord.x;
    if(pnt == tet.y) return coord.y;
    if(pnt == tet.z) return coord.z;
    return coord.w;
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
	
	const uint iPair = bodyPairHash[ind].value;
	if(srcPairs[iPair].x == bodyPairHash[ind].key) {
	    splits[iPair].x = ind;
	}
	else {
	    splits[iPair].y = ind;
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
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float * mass,
	                                    uint4 * indices,
	                                    uint * pointStart,
	                                    uint * indexStart,
	                                    uint * bodyCount, 
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iPair = ind>>1;
	const int isRgt = (ind & 1);
	
	uint4 ia;
	uint dstInd;
	if(isRgt) {
	    dstInd = splits[iPair].y;
	    ia = computePointIndex(pointStart, indexStart, indices, pairs[iPair].y);
	}
	else {
	    dstInd = splits[iPair].x;
	    ia = computePointIndex(pointStart, indexStart, indices, pairs[iPair].x);
	}
	
	uint n = getBodyCountAt(dstInd, bodyCount);
	
	invMass[dstInd] = (float)n / (absoluteValueF(mass[ia.x]) + absoluteValueF(mass[ia.y]) + absoluteValueF(mass[ia.z]) + absoluteValueF(mass[ia.w]));
}

__global__ void setContactConstraint_kernel(ContactConstraint* constraints,
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
    __shared__ float3 sVel[SETCONSTRAINT_TPB];
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint iContact = ind>>1;
	
	ContactData contact = contacts[iContact];
	
	int isRgt = (threadIdx.x & 1);
	
	uint4 ia;
	if(isRgt>0) {
	    ia = computePointIndex(pointStarts, indexStarts, indices, pairs[iContact].y);
	    constraints[iContact].coordB = localCoordinate(ia, srcPos, contact.localB);
	    interpolate_float3i(sVel[threadIdx.x], ia, srcVel, &constraints[iContact].coordB);
	    __syncthreads();
	}
	else {
	    ia = computePointIndex(pointStarts, indexStarts, indices, pairs[iContact].x);
	    constraints[iContact].coordA = localCoordinate(ia, srcPos, contact.localA);
	    interpolate_float3i(sVel[threadIdx.x], ia, srcVel, &constraints[iContact].coordA);
	    __syncthreads();
	}

	if(isRgt) return;
	
	constraints[iContact].lambda = 0.f;

	const uint2 dstInd = splits[iContact];
	
	float3 nA = normalOnA(contact);
	float3 nB = float3_reverse(nA);
	float3 torqueA = float3_cross(contact.localA, nA);
	float3 torqueB = float3_cross(contact.localB, nB);
	
	constraints[iContact].normal = nA;// float3_from_float4(contact.separateAxis);
	constraints[iContact].Minv = computeMassTensor(nA, nB, contact.localA, contact.localB,
	                            torqueA, torqueB,
	                            splitMass[dstInd.x], splitMass[dstInd.y]);
	
	float rel = computeRelativeVelocity1(nA, nB,
	                        sVel[threadIdx.x], sVel[threadIdx.x+1]);
	if(rel * rel < 0.01f) rel = 0.f;
	constraints[iContact].relVel = rel;
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

__global__ void solveContactWoJ_kernel(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * pointStarts,
                        uint * indexStarts,
                        uint maxInd)
{
    __shared__ float3 sVel[SOLVECONTACT_TPB];
    __shared__ float3 sN[SOLVECONTACT_TPB];
    
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iContact = ind>>1;
	
	uint splitInd = splits[iContact].x;
	uint iBody = pairs[iContact].x;
	BarycentricCoordinate coord = constraints[iContact].coordA;
	
	if((threadIdx.x & 1)>0) {
	    splitInd = splits[iContact].y;
	    iBody = pairs[iContact].y;
	    coord = constraints[iContact].coordB;
	}

// initial velocities
    uint4 ia = computePointIndex(pointStarts, indexStarts, indices, iBody);
    
    float3 velA;
    interpolate_float3i(velA, ia, velocities, &coord);
    
	float3 nA = constraints[iContact].normal;
	float3 rA = contacts[iContact].localA;
	
	if((ind & 1)>0) {
	    nA = float3_reverse(nA);
	    rA = contacts[iContact].localB;
	}
	
// N pointing inside object
// T = r X N	
	float3 torqueA = float3_cross(rA, nA);
	
	addDeltaVelocity(velA, 
        deltaLinearVelocity[splitInd],
        deltaAngularVelocity[splitInd],
        nA, rA,
        &coord);
    
    sN[threadIdx.x] = nA;
    sVel[threadIdx.x] = velA;
    __syncthreads();
    
    uint iLeft = (threadIdx.x>>1)<<1;
    uint iRight = iLeft + 1;
	
	float J = computeRelativeVelocity1(sN[iLeft], sN[iRight],
	                        sVel[iLeft], sVel[iRight]);

	J += constraints[iContact].relVel;
	J *= constraints[iContact].Minv;
	
	float prevSum = constraints[iContact].lambda;
	float updated = prevSum;
	updated += J;
	if(updated < 0.f) updated = 0.f;
	
	if((threadIdx.x & 1)==0) constraints[iContact].lambda = updated;
	
	J = updated - prevSum;
	
	const float invMassA = splitMass[splitInd];
	
    if(invMassA > 1e-10f)
        applyImpulse(deltaLinearVelocity[splitInd], J * invMassA, nA);
	//applyImpulse(deltaAngularVelocity[splitInd], J * invMassA, torqueA);
}

__global__ void solveContact_kernel(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * pointStarts,
                        uint * indexStarts,
                        uint maxInd,
                        float * deltaJ,
                        int maxNIt,
                        int it)
{
    __shared__ float3 sVel[SOLVECONTACT_TPB];
    __shared__ float3 sN[SOLVECONTACT_TPB];
    
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iContact = ind>>1;
	
	uint splitInd = splits[iContact].x;
	uint iBody = pairs[iContact].x;
	BarycentricCoordinate coord = constraints[iContact].coordA;
	
	if((threadIdx.x & 1)>0) {
	    splitInd = splits[iContact].y;
	    iBody = pairs[iContact].y;
	    coord = constraints[iContact].coordB;
	}

// initial velocities
    uint4 ia = computePointIndex(pointStarts, indexStarts, indices, iBody);
    
    float3 velA;
    interpolate_float3i(velA, ia, velocities, &coord);
    
	float3 nA = constraints[iContact].normal;
	float3 rA = contacts[iContact].localA;
	
	if((ind & 1)>0) {
	    nA = float3_reverse(nA);
	    rA = contacts[iContact].localB;
	}
	
// N pointing inside object
// T = r X N	
	float3 torqueA = float3_cross(rA, nA);
	
	addDeltaVelocity(velA, 
        deltaLinearVelocity[splitInd],
        deltaAngularVelocity[splitInd],
        nA, rA,
        &coord);
    
    sN[threadIdx.x] = nA;
    sVel[threadIdx.x] = velA;
    __syncthreads();
    
    uint iLeft = (threadIdx.x>>1)<<1;
    uint iRight = iLeft + 1;
	
	float J = computeRelativeVelocity1(sN[iLeft], sN[iRight],
	                        sVel[iLeft], sVel[iRight]);

	J += constraints[iContact].relVel;
	J *= constraints[iContact].Minv;
	
	float prevSum = constraints[iContact].lambda;
	float updated = prevSum;
	updated += J;
	if(updated < 0.f) updated = 0.f;
	
	if((threadIdx.x & 1)==0) constraints[iContact].lambda = updated;
	
	J = updated - prevSum;
	
	if((threadIdx.x & 1)==0) deltaJ[iContact * maxNIt + it] = J;
	
	const float invMassA = splitMass[splitInd];
	
	applyImpulse(deltaLinearVelocity[splitInd], J * invMassA, nA);
	applyImpulse(deltaAngularVelocity[splitInd], J * invMassA, torqueA);
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
        pntTetHash[istart + 1].key = 1<<30;
        pntTetHash[istart + 2].key = 1<<30;
        pntTetHash[istart + 3].key = 1<<30;
        return;
	}
	   
	const unsigned iContact = ind>>1;
	
	uint splitInd = splits[iContact].x;
	uint iBody = pairs[iContact].x;
	
	if((ind & 1)>0) {
	    splitInd = splits[iContact].y;
	    iBody = pairs[iContact].y;
	}
	    
	if(bodyCount[splitInd] < 1) {
// redundant
        pntTetHash[istart].key = 1<<30;
        pntTetHash[istart + 1].key = 1<<30;
        pntTetHash[istart + 2].key = 1<<30;
        pntTetHash[istart + 3].key = 1<<30;
	}
	else {
	    const uint4 ia = computePointIndex(pointStart, indexStart, indices, iBody);
	    
	    pntTetHash[istart  ].key = ia.x;
	    pntTetHash[istart  ].value = ind;
	    pntTetHash[istart+1].key = ia.y;
	    pntTetHash[istart+1].value = ind;
	    pntTetHash[istart+2].key = ia.z;
	    pntTetHash[istart+2].value = ind;
	    pntTetHash[istart+3].key = ia.w;
	    pntTetHash[istart+3].value = ind;
	}
}

__global__ void updateVelocity_kernel(float3 * dstVelocity,
                    float3 * deltaLinearVelocity,
	                float3 * deltaAngularVelocity,
	                KeyValuePair * pntTetHash,
                    uint2 * pairs,
                    uint2 * splits,
                    ContactConstraint * constraints,
                    ContactData * contacts,
                    float3 * position,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iPnt = pntTetHash[ind].key;
	
	if(iPnt > (1<<24)) return;
	
	int isFirst = 0;
	
	if(ind < 1) isFirst = 1;
	else if(pntTetHash[ind - 1].key != iPnt) isFirst = 1;
	
	if(!isFirst) return;
	
	float3 sumLinVel = make_float3(0.f, 0.f, 0.f);
	float count = 0.f;
	uint cur = ind;
	uint iContact, splitInd, iBody;
	uint4 iTet;
	float weight;
	BarycentricCoordinate coord;
	float3 r, normal;
#if ENABLE_DEFORMABILITY
    float3 vRot;
#endif
	for(;;) {
	    iContact = pntTetHash[cur].value>>1;
	
	    splitInd = splits[iContact].x;
	    iBody = pairs[iContact].x;
	    coord = constraints[iContact].coordA;
	    r = contacts[iContact].localA;
	    normal = constraints[iContact].normal;
	
        if((pntTetHash[cur].value & 1)>0) {
            splitInd = splits[iContact].y;
            iBody = pairs[iContact].y;
            coord = constraints[iContact].coordB;
            r = contacts[iContact].localB;
            normal = float3_reverse(normal);
        }
        
        sumLinVel = float3_add(sumLinVel, deltaLinearVelocity[splitInd]);
        
        iTet = computePointIndex(objectPointStarts, objectIndexStarts, indices, iBody);
        weight = getPntTetWeight(iPnt, iTet, coord);
#if ENABLE_DEFORMABILITY
        deformMotion(vRot, r, normal, deltaAngularVelocity[splitInd]);
// weighted by vex coord        
        vRot = scale_float3_by(vRot, weight);

        sumLinVel = float3_add(sumLinVel, vRot);
#endif
        count += 1.f;
        
        if(cur == maxInd - 1) break;
	    cur++;
	    if(pntTetHash[cur].key != iPnt) break;
	}
	
	if(count > 1.f) sumLinVel = scale_float3_by(sumLinVel, 1.f / count);
	
	dstVelocity[iPnt] = float3_add(dstVelocity[iPnt], sumLinVel);
	// dstVelocity[iPnt] = float3_add(dstVelocity[iPnt], deltaLinearVelocity[splitInd]);
	// float3_set_zero(dstVelocity[iPnt]);
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
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float * mass,
	                                    uint4 * ind,
	                                    uint * perObjPointStart,
	                                    uint * perObjectIndexStart,
                                        uint * bodyCount, 
                                        uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    
    computeSplitInvMass_kernel<<< grid, block >>>(invMass,
                                        splits,
                                        pairs,
                                        mass,
	                                    ind,
	                                    perObjPointStart,
	                                    perObjectIndexStart,
	                                    bodyCount, 
	                                    bufLength);
}

void simpleContactSolverSetContactConstraint(ContactConstraint* constraints,
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        uint4 * ind,
                                        uint * perObjPointStart,
                                        uint * perObjectIndexStart,
                                        float * splitMass,
                                        ContactData * contacts,
                                        uint numContacts2)
{
    dim3 block(SETCONSTRAINT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SETCONSTRAINT_TPB);
    dim3 grid(nblk, 1, 1);
    
    setContactConstraint_kernel<<< grid, block >>>(constraints,
                                        splits,
                                        pairs,
                                        pos,
                                        vel,
                                        ind,
                                        perObjPointStart,
                                        perObjectIndexStart,
                                        splitMass,
                                        contacts,
                                        numContacts2);
    // cudaDeviceSynchronize();
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

void simpleContactSolverSolveContactWoJ(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * perObjPointStart,
                        uint * perObjectIndexStart,
                        uint numContacts2)
{
    dim3 block(SOLVECONTACT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SOLVECONTACT_TPB);
    dim3 grid(nblk, 1, 1);
    
    solveContactWoJ_kernel<<< grid, block >>>(constraints,
                        deltaLinearVelocity,
	                    deltaAngularVelocity,
	                    pairs,
                        splits,
	                    splitMass,
	                    contacts,
	                    positions,
                        velocities,
                        indices,
                        perObjPointStart,
                        perObjectIndexStart,
                        numContacts2);
}

void simpleContactSolverSolveContact(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
                        uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * perObjPointStart,
                        uint * perObjectIndexStart,
                        uint numContacts2,
                        float * deltaJ,
                        int maxNIt,
                        int it)
{
    dim3 block(SOLVECONTACT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SOLVECONTACT_TPB);
    dim3 grid(nblk, 1, 1);
    
    solveContact_kernel<<< grid, block >>>(constraints,
                        deltaLinearVelocity,
	                    deltaAngularVelocity,
	                    pairs,
                        splits,
	                    splitMass,
	                    contacts,
	                    positions,
                        velocities,
                        indices,
                        perObjPointStart,
                        perObjectIndexStart,
                        numContacts2,
                        deltaJ,
                        maxNIt,
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

void simpleContactSolverUpdateVelocity(float3 * dstVelocity,
                    float3 * deltaLinearVelocity,
	                float3 * deltaAngularVelocity,
	                KeyValuePair * pntTetHash,
                    uint2 * pairs,
                    uint2 * splits,
                    ContactConstraint * constraints,
                    ContactData * contacts,
                    float3 * position,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numPoints)
{
    uint tpb = 256;

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numPoints, tpb);
    dim3 grid(nblk, 1, 1);
    
    updateVelocity_kernel<<< grid, block >>>(dstVelocity,
                    deltaLinearVelocity,
	                deltaAngularVelocity,
	                pntTetHash,
                    pairs,
                    splits,
                    constraints,
                    contacts,
                    position,
                    indices,
                    objectPointStarts,
                    objectIndexStarts,
                    numPoints);
}

}
