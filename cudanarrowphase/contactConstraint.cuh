#ifndef CONTACTCONSTRAINT_CUH
#define CONTACTCONSTRAINT_CUH

#include "contactSolverCommon.cuh"

inline __device__ BarycentricCoordinate localCoordinate1(uint4 ia,
                                    float3 * position,
                                    float3 localP)
{
    float3 q;
	float3_average4(q, position, ia);
	float3_add_inplace(q, localP);
	return getBarycentricCoordinate4i(q, position, ia);
}

// pointing from B to A
inline __device__ float3 normalOnA(const ContactData & contact)
{
    return float3_normalize(float3_from_float4(contact.separateAxis));
}

inline __device__ float computeMassTensor(//float3 nA, float3 nB, 
                                       // float3 rA, float3 rB,
                                       // float3 torqueA, float3 torqueB,
                                        float invMassA, float invMassB)
{
    //float3 jmjA = float3_cross( scale_float3_by(torqueA, invMassA), rA );
    //float3 jmjB = float3_cross( scale_float3_by(torqueB, invMassB), rB );
    
    return -1.f/(invMassA + invMassB);    
         // invMassA * float3_dot(jmjA, nA) + 
         // invMassB * float3_dot(jmjB, nB));
}

inline __device__ float computeRelativeVelocityLinearOnly(float3 nA,
                            float3 nB,
                            float3 linearVelocityA, 
                            float3 linearVelocityB)
{
    return float3_dot(linearVelocityA, nA) +
            float3_dot(linearVelocityB, nB);
}

__global__ void prepareNoPenetratingContactConstraint_kernel(ContactConstraint* constraints,
                            float3 * contactLinearVel,
                            uint2 * splits,
                            float3 * srcPos,
                            float3 * srcVel,
                            float3 * srcImpulse,
                            float * splitMass,
                            ContactData * contacts,
                            uint4 * tetind,
                            uint maxInd)
{
    __shared__ float4 sCoord[SETCONSTRAINT_TPB];
    __shared__ float3 sVel[SETCONSTRAINT_TPB];
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint iContact = ind>>1;
	int isRgt = (threadIdx.x & 1);
	   
    ContactData contact = contacts[iContact];
	
    ContactConstraint outConstraint;
	
	BarycentricCoordinate wei;
    float3 v0, v1;
	uint4 ia = tetind[ind];
	if(isRgt>0) {
	    wei = localCoordinate1(ia, srcPos, contact.localB);
	    
	}
	else {
	    wei = localCoordinate1(ia, srcPos, contact.localA);
	}
	sCoord[threadIdx.x] = wei;
	interpolate_float3i(v0, ia, srcVel, &wei);
    interpolate_float3i(v1, ia, srcImpulse, &wei);
    sVel[threadIdx.x] = float3_add(v0, v1);
	contactLinearVel[ind] = sVel[threadIdx.x];
	__syncthreads();

	if(isRgt) return;
	
	outConstraint.lambda = 0.f;
	
	const uint2 dstInd = splits[iContact];
	
	float3 nA = normalOnA(contact);
	float3 nB = float3_reverse(nA);
	//float3 torqueA = float3_cross(contact.localA, nA);
	//float3 torqueB = float3_cross(contact.localB, nB);
	
	outConstraint.normal = nA;// float3_from_float4(contact.separateAxis);
	outConstraint.Minv = computeMassTensor(//nA, nB, contact.localA, contact.localB,
	                            //torqueA, torqueB,
	                            splitMass[dstInd.x], splitMass[dstInd.y]);
	
	//if(splitMass[dstInd.x] > 1e-5f) 
	//    sVel[threadIdx.x].y += VYACCELERATION;
	//if(splitMass[dstInd.y] > 1e-5f) 
	//    sVel[threadIdx.x+1].y += VYACCELERATION;

	float rel = computeRelativeVelocityLinearOnly(nA, nB,
	                        sVel[threadIdx.x], sVel[threadIdx.x+1]);
// penalty for shallow penetrations
    if(contact.timeOfImpact < 1e-9f) {
        rel -= 4.f + float3_length( float3_difference( sVel[threadIdx.x], sVel[threadIdx.x+1] ) );
    }
    outConstraint.coordA = sCoord[threadIdx.x];
    outConstraint.coordB = sCoord[threadIdx.x+1];
	outConstraint.relVel = rel;
	constraints[iContact] = outConstraint;
}

#endif        //  #ifndef CONTACTCONSTRAINT_CUH
