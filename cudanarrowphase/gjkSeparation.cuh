#ifndef GJKSEPARATION_CUH
#define GJKSEPARATION_CUH

#include <gjk_math.cu>
#include <stripedModel.cuh>

#define GJK_BLOCK_SIZE 64

inline __device__ float velocityOnTetrahedronAlong2(float3 * v, const BarycentricCoordinate & coord, const float3 & d)
{
    float3 vot = make_float3(0.f, 0.f, 0.f);
    if(coord.x > 1e-5f)
        float3_add_inplace(vot, scale_float3_by(v[0], coord.x));
    if(coord.y > 1e-5f)
        float3_add_inplace(vot, scale_float3_by(v[2], coord.y));
    if(coord.z > 1e-5f)
        float3_add_inplace(vot, scale_float3_by(v[4], coord.z));
    if(coord.w > 1e-5f)
        float3_add_inplace(vot, scale_float3_by(v[6], coord.w));
    
    return float3_dot(vot, d);
}

__global__ void computeInitialSeparation_kernel(ContactData * dstContact,
    float3 * pos,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

// assuming already penetrated
	dstContact[ind].separateAxis=make_float4(0.f, 0.f, 0.f, 0.f);
	dstContact[ind].timeOfImpact = - GJK_STEPSIZE;
        
	float3 * ppos = & pos[ind<<3];
	
	sPrxA[threadIdx.x].p[0] = ppos[0];
	sPrxB[threadIdx.x].p[0] = ppos[1];
	sPrxA[threadIdx.x].p[1] = ppos[2];
	sPrxB[threadIdx.x].p[1] = ppos[3];
	sPrxA[threadIdx.x].p[2] = ppos[4];
	sPrxB[threadIdx.x].p[2] = ppos[5];
	sPrxA[threadIdx.x].p[3] = ppos[6];
	sPrxB[threadIdx.x].p[3] = ppos[7];
	
	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	float4 sas;
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
	    coord);
//  intersected try zero margin	
/*	if(sas.w < 1.f) {
	    computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], 0.f, ctc, 
		sas,
	    coord);
	}*/
// still intersected negative toi
    if(sas.w < 1.f)
        return;

// output	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
	dstContact[ind].separateAxis = sas;
	dstContact[ind].timeOfImpact = 1e-6f;
}

__global__ void advanceTimeOfImpactIterative_kernel(ContactData * dstContact,
    float3 * pos, float3 * vel, 
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	const ContactData ct = dstContact[ind];
	
// already determined no contact
	if(ct.separateAxis.w < 1.f || ct.timeOfImpact > GJK_STEPSIZE)
	    return;
	
	float3 * ppos = & pos[ind<<3];
	float3 * pvel = & vel[ind<<3];
	
	float4 sas = ct.separateAxis;
	
	const float3 nor = float3_normalize(float3_from_float4(sas));
	
	float closeInSpeed = velocityOnTetrahedronAlong2(&pvel[1], getBarycentricCoordinate4Relative2(ct.localB, &ppos[1]), 
	                                                nor)
	                    - velocityOnTetrahedronAlong2(pvel, getBarycentricCoordinate4Relative2(ct.localA, ppos), 
	                                                nor);
// going apart     
    if(closeInSpeed < 1e-8f) { 
        dstContact[ind].timeOfImpact = 1e8f;
        return;
    }
    
	float separateDistance = float4_length(ct.separateAxis);
	
// within thin shell margin
	if(separateDistance <= GJK_THIN_MARGIN2)
	    return;
	
// use thin shell margin
	// separateDistance -= GJK_THIN_MARGIN2;

	const float toi = ct.timeOfImpact + separateDistance / closeInSpeed * .51f;

// too far away	
	if(toi > GJK_STEPSIZE) { 
        dstContact[ind].timeOfImpact = 1e8f;
        return;
    }
	
	sPrxA[threadIdx.x].p[0] = float3_add( ppos[0], scale_float3_by(pvel[0], toi) );
	sPrxB[threadIdx.x].p[0] = float3_add( ppos[1], scale_float3_by(pvel[1], toi) );
	sPrxA[threadIdx.x].p[1] = float3_add( ppos[2], scale_float3_by(pvel[2], toi) );
	sPrxB[threadIdx.x].p[1] = float3_add( ppos[3], scale_float3_by(pvel[3], toi) );
	sPrxA[threadIdx.x].p[2] = float3_add( ppos[4], scale_float3_by(pvel[4], toi) );
	sPrxB[threadIdx.x].p[2] = float3_add( ppos[5], scale_float3_by(pvel[5], toi) );
	sPrxA[threadIdx.x].p[3] = float3_add( ppos[6], scale_float3_by(pvel[6], toi) );
	sPrxB[threadIdx.x].p[3] = float3_add( ppos[7], scale_float3_by(pvel[7], toi) );
	
	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
            coord); 
// penetrated use result of last step       
    if(sas.w < 1.f) return;
 
// output
    interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
	dstContact[ind].separateAxis = sas;
	dstContact[ind].timeOfImpact = toi;
}


__global__ void separateShallowPenetration_kernel(ContactData * dstContact,
    float3 * pos,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	if(dstContact[ind].timeOfImpact > 0.f) return;

// assuming already penetrated
	dstContact[ind].separateAxis=make_float4(0.f, 0.f, 0.f, 0.f);
	
	float3 * ppos = & pos[ind<<3];
	
	sPrxA[threadIdx.x].p[0] = ppos[0];
	sPrxB[threadIdx.x].p[0] = ppos[1];
	sPrxA[threadIdx.x].p[1] = ppos[2];
	sPrxB[threadIdx.x].p[1] = ppos[3];
	sPrxA[threadIdx.x].p[2] = ppos[4];
	sPrxB[threadIdx.x].p[2] = ppos[5];
	sPrxA[threadIdx.x].p[3] = ppos[6];
	sPrxB[threadIdx.x].p[3] = ppos[7];
	
	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	float4 sas;
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], 0.f, ctc, sas, 
	    coord);

// still intersected negative toi
    if(sas.w < 1.f)
        return;

// output	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
	dstContact[ind].separateAxis = sas;
	dstContact[ind].timeOfImpact = 1e-10f;
}

#endif        //  #ifndef GJKSEPARATION_CUH

