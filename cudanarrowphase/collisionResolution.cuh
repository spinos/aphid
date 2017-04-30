#ifndef COLLISIONRESOLUTION_CUH
#define COLLISIONRESOLUTION_CUH

#include "contactSolverCommon.cuh"

inline __device__ void addDeltaVelocityLinearOnly(float3 & dst, 
        float3 deltaLinearVelocity)
{
    float3_add_inplace(dst, deltaLinearVelocity);
}

inline __device__ void applyImpulse(float3 & dst, float J, float3 N)
{
    float3_add_inplace(dst, scale_float3_by(N, J));
}

__global__ void resolveCollision_kernel(ContactConstraint* constraints,
                        float3 * contactLinearVelocity,
                        float3 * deltaLinearVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    uint maxInd)
{
    __shared__ float3 sVel[SOLVECONTACT_TPB];
    __shared__ float3 sN[SOLVECONTACT_TPB];
    
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iContact = ind>>1;
	
	const ContactConstraint inConstraint = constraints[iContact];
	
	uint splitInd;
	
	if((threadIdx.x & 1)>0) {
	    splitInd = splits[iContact].y;
	}
	else {
	    splitInd = splits[iContact].x;
	}

// initial velocities
    float3 velA = contactLinearVelocity[ind];
    
    const float invMassA = splitMass[splitInd];
	//if(invMassA > 1e-5f)
	//     velA.y += VYACCELERATION;
    
	float3 nA = inConstraint.normal;
	
	if((ind & 1)>0) {
	    nA = float3_reverse(nA);
	}
	
// N pointing inside object
// T = r X N	
	//float3 torqueA = float3_cross(rA, nA);
	
	addDeltaVelocityLinearOnly(velA, 
        deltaLinearVelocity[splitInd]);
    
    sN[threadIdx.x] = nA;
    sVel[threadIdx.x] = velA;
    __syncthreads();
    
    uint iLeft = (threadIdx.x>>1)<<1;
    uint iRight = iLeft + 1;
	
	float J = computeRelativeVelocity1(sN[iLeft], sN[iRight],
	                        sVel[iLeft], sVel[iRight]);

/*
 *  reference
 *  Game Physics 
 *  Game and Media Technology 
 *  Master Program - Utrecht University 
 *  Dr. Nicolas Pronost 
 *  Lecture 7 Collision Resolution Pg. 18
 *  j = (1 + Cr)Vr.N*M^-1
 *  Cr is restitution
 */
    float restitution = .8f;
    //if(J * J < 0.01f) restitution = 0.f;
    
    J += restitution * inConstraint.relVel;
    J *= inConstraint.Minv;
	
	float prevSum = constraints[iContact].lambda;
	float updated = prevSum;
	updated += J;
    if(updated < 0.f) updated = 0.f;
	
	if((threadIdx.x & 1)==0) constraints[iContact].lambda = updated;
	
	J = updated - prevSum;
	
	applyImpulse(deltaLinearVelocity[splitInd], J * invMassA, nA);
}

__global__ void resolveFriction_kernel(ContactConstraint* constraints,
                        float3 * contactLinearVelocity,
                        float3 * deltaLinearVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    uint maxInd)
{
    __shared__ float3 sVel[SOLVECONTACT_TPB];
    __shared__ float3 sN[SOLVECONTACT_TPB];
    
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint iContact = ind>>1;
	
	const ContactConstraint inConstraint = constraints[iContact];
	
	uint splitInd;
	
	if((threadIdx.x & 1)>0) {
	    splitInd = splits[iContact].y;
	}
	else {
	    splitInd = splits[iContact].x;
	}

// initial velocities
    float3 velA = contactLinearVelocity[ind];
    
    const float invMassA = splitMass[splitInd];

	float3 nA = inConstraint.normal;
	
	if((ind & 1)>0) {
	    nA = float3_reverse(nA);
	}
	
	addDeltaVelocityLinearOnly(velA, 
        deltaLinearVelocity[splitInd]);
    
    sN[threadIdx.x] = nA;
    sVel[threadIdx.x] = velA;
    __syncthreads();
    
    uint iLeft = (threadIdx.x>>1)<<1;
    uint iRight = iLeft + 1;

// VA - VB
	float3 velRel = float3_difference(sVel[iLeft], sVel[iRight]);

    if((ind & 1)>0) {
	    float3_scale_inplace(velRel, -1.f);
	}
	
	float speed = float3_length(velRel);
	if(speed < 1e-6f) return;

// direction of relative velocity
	float3_divide_inplace(velRel, speed);
	
	float ang = float3_dot(velRel, nA);
	if(ang < 0.f) ang = - ang;

// project to contact normal	
	float3_scale_inplace(nA, ang);

// inversed direction of rectional impulse
// will multiply negative mass tensor bellow	
	float3 reaction = float3_difference(velRel, nA);
	
	float mag = float3_length(reaction);
	if(mag < 1e-6f) return;

	float3_divide_inplace(reaction, mag);

// large enough to be dynamic friction	
	if(mag > .439f) mag = .439f;

    float3_scale_inplace(reaction, mag * speed * inConstraint.Minv * invMassA);
	float3_add_inplace(deltaLinearVelocity[splitInd], reaction);
}

#endif        //  #ifndef COLLISIONRESOLUTION_CUH
