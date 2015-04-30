#include "narrowphase_implement.h"
#include <bvh_math.cu>
#include <gjk_math.cu>
#include <stripedModel.cu>
#include <CudaBase.h>

#define GJK_BLOCK_SIZE 64

inline __device__ float totalSpeed(float3 * vel, const uint4 & ia, const uint4 & ib)
{
    return (float3_length2(vel[ia.x]) + float3_length2(vel[ia.y]) +
            float3_length2(vel[ia.z]) + float3_length2(vel[ia.w]) +
            float3_length2(vel[ib.x]) + float3_length2(vel[ib.y]) +
            float3_length2(vel[ib.z]) + float3_length2(vel[ib.w]));
}

inline __device__ void t0Tetrahedron(TetrahedronProxy & prx, 
                                            const uint4 & v, 
                                            float3 * pos)
{
    prx.p[0] = pos[v.x];
    prx.p[1] = pos[v.y];
    prx.p[2] = pos[v.z];
    prx.p[3] = pos[v.w];
}

inline __device__ void progressTetrahedron(TetrahedronProxy & prx, 
                                            const uint4 & v, 
                                            float3 * pos,
                                            float3 * vel,
                                            float h)
{
    prx.p[0] = float3_add(pos[v.x], scale_float3_by(vel[v.x], h));
    prx.p[1] = float3_add(pos[v.y], scale_float3_by(vel[v.y], h));
    prx.p[2] = float3_add(pos[v.z], scale_float3_by(vel[v.z], h));
    prx.p[3] = float3_add(pos[v.w], scale_float3_by(vel[v.w], h));
}

inline __device__ float velocityOnTetrahedronAlong(float3 * v, const uint4 & t, const BarycentricCoordinate & coord, const float3 & d)
{
    float3 vot = make_float3(0.f, 0.f, 0.f);
    if(coord.x > 1e-5f)
        vot = float3_add(vot, scale_float3_by(v[t.x], coord.x));
    if(coord.y > 1e-5f)
        vot = float3_add(vot, scale_float3_by(v[t.y], coord.y));
    if(coord.z > 1e-5f)
        vot = float3_add(vot, scale_float3_by(v[t.z], coord.z));
    if(coord.w > 1e-5f)
        vot = float3_add(vot, scale_float3_by(v[t.w], coord.w));
    
    return float3_dot(vot, d);
}

__global__ void computeSeparateAxis_kernel(ContactData * dstContact,
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4* indices, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	uint4 ita = computePointIndex(pointStart, indexStart, indices, pairs[ind].x);
	uint4 itb = computePointIndex(pointStart, indexStart, indices, pairs[ind].y);
	
	progressTetrahedron(sPrxA[threadIdx.x], ita, pos, vel, 0.01667f);
	progressTetrahedron(sPrxB[threadIdx.x], itb, pos, vel, 0.01667f);

	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, dstContact[ind].separateAxis, 
	    coord);
	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
}

__global__ void computeTimeOfImpact_kernel(ContactData * dstContact,
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4 * indices, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	dstContact[ind].separateAxis=make_float4(0.f, 0.f, 0.f, 0.f);
	dstContact[ind].timeOfImpact = 1e8;
	
	const uint4 ita = computePointIndex(pointStart, indexStart, indices, pairs[ind].x);
	const uint4 itb = computePointIndex(pointStart, indexStart, indices, pairs[ind].y);
	
	if(totalSpeed(vel, ita, itb) < 1e-8) return;
	
	t0Tetrahedron(sPrxA[threadIdx.x], ita, pos);
	t0Tetrahedron(sPrxB[threadIdx.x], itb, pos);

	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	float4 sas;
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
	    coord);
// intersected try zero margin	
	if(sas.w < 1.f) {
	    computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], 0.f, ctc, sas, 
	    coord);
	}
// still intersected no solution
	if(sas.w < 1.f) return;
	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
	
	float3 nor = float3_normalize(float3_from_float4(sas));
	
	float closeInSpeed = velocityOnTetrahedronAlong(vel, itb, getBarycentricCoordinate4Relativei(dstContact[ind].localB, pos, itb), 
	                                                nor)
	                    - velocityOnTetrahedronAlong(vel, ita, getBarycentricCoordinate4Relativei(dstContact[ind].localA, pos, ita), 
	                                                nor);
// going apart no contact     
    if(closeInSpeed < 1e-8) { 
        return;
    }
	
	float separateDistance = float4_length(sas);
// within thin shell margin
	if(separateDistance < GJK_THIN_MARGIN2) {
	    dstContact[ind].timeOfImpact = 1e-9;
	    dstContact[ind].separateAxis = sas;
        return;
	}

// use thin shell margin
	separateDistance -= GJK_THIN_MARGIN2;
	
	dstContact[ind].separateAxis = sas;
	
	float lastDistance = separateDistance;
	float toi = 0.f;
	int i = 0;
    while (i<GJK_MAX_NUM_ITERATIONS) {
// going apart       
        if(closeInSpeed < 1e-8) { 
            dstContact[ind].timeOfImpact = 1e8;
            
// for debug purpose
            // dstContact[ind].separateAxis = sas;
            // interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        
            break;
        }
        
        toi += separateDistance / closeInSpeed * .743f;
// too far away       
        if(toi > GJK_STEPSIZE) { 
            dstContact[ind].timeOfImpact = toi;
            
// for debug purpose
            // dstContact[ind].separateAxis = sas;
            // interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        
            break;   
        }
        
        progressTetrahedron(sPrxA[threadIdx.x], ita, pos, vel, toi);
        progressTetrahedron(sPrxB[threadIdx.x], itb, pos, vel, toi);
  
        computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
            coord); 
// penetrated use result of last step       
        if(sas.w < 1.f) { 
            break;
        }
        
// output toi and r
        dstContact[ind].timeOfImpact = toi;
        interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);

        separateDistance = float4_length(sas);
// close enough use result of last step
        if(separateDistance < 0.001f) { 
            break;
        }
        
// going apart, no contact
        if(separateDistance >= lastDistance) {
            dstContact[ind].timeOfImpact = 1e8;
            break;
        }
        
        lastDistance = separateDistance;
        
// output sa
        dstContact[ind].separateAxis = sas;
        
        nor = float3_normalize(float3_from_float4(sas));
        closeInSpeed = velocityOnTetrahedronAlong(vel, itb, getBarycentricCoordinate4Relativei(dstContact[ind].localB, pos, itb), 
	                                                nor)
	                    - velocityOnTetrahedronAlong(vel, ita, getBarycentricCoordinate4Relativei(dstContact[ind].localA, pos, ita), 
	                                                nor);
        
        i++;
    }
}

__global__ void advanceTimeOfImpactIterative_kernel(ContactData * dstContact,
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4 * indices, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

// already determined no contact
	if(dstContact[ind].separateAxis.w < 1.f || dstContact[ind].timeOfImpact > GJK_STEPSIZE)
	    return;
	
	const uint4 ita = computePointIndex(pointStart, indexStart, indices, pairs[ind].x);
	const uint4 itb = computePointIndex(pointStart, indexStart, indices, pairs[ind].y);
	
	float4 sas = dstContact[ind].separateAxis;
	
	float3 nor = float3_normalize(float3_from_float4(sas));
	
	float closeInSpeed = velocityOnTetrahedronAlong(vel, itb, getBarycentricCoordinate4Relativei(dstContact[ind].localB, pos, itb), 
	                                                nor)
	                    - velocityOnTetrahedronAlong(vel, ita, getBarycentricCoordinate4Relativei(dstContact[ind].localA, pos, ita), 
	                                                nor);
// going apart     
    if(closeInSpeed < 1e-8f) { 
        dstContact[ind].timeOfImpact = 1e8f;
        return;
    }
    
	float separateDistance = float4_length(dstContact[ind].separateAxis);
	
// within thin shell margin
	if(separateDistance < GJK_THIN_MARGIN2)
	    return;
	
// use thin shell margin
	separateDistance -= GJK_THIN_MARGIN2;

	float toi = dstContact[ind].timeOfImpact + separateDistance / closeInSpeed * .732f;

// too far away	
	if(toi > GJK_STEPSIZE) { 
        dstContact[ind].timeOfImpact = 1e8f;
        return;
    }
	
	progressTetrahedron(sPrxA[threadIdx.x], ita, pos, vel, toi);
	progressTetrahedron(sPrxB[threadIdx.x], itb, pos, vel, toi);
	
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

__global__ void computeInitialSeparation_kernel(ContactData * dstContact,
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4 * indices, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	dstContact[ind].separateAxis=make_float4(0.f, 0.f, 0.f, 0.f);
	dstContact[ind].timeOfImpact = 1e8f;
	
	const uint4 ita = computePointIndex(pointStart, indexStart, indices, pairs[ind].x);
	const uint4 itb = computePointIndex(pointStart, indexStart, indices, pairs[ind].y);
	
	t0Tetrahedron(sPrxA[threadIdx.x], ita, pos);
	t0Tetrahedron(sPrxB[threadIdx.x], itb, pos);

	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	float4 sas;
//	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
//	    coord);
// intersected try zero margin	
//	if(sas.w < 1.f) {
	    computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], 0.f, ctc, sas, 
	    coord);
//	}
// still intersected no solution
	if(sas.w < 1.f) return;

// output	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
	dstContact[ind].separateAxis = sas;
	dstContact[ind].timeOfImpact = 1e-9f;
}

__global__ void computeValidPairs_kernel(uint* dstCounts, 
                    ContactData * srcContact, 
                    uint numContacts, 
                    uint scanBufferLength)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= scanBufferLength) return;
	
	dstCounts[ind] = 0;
	if(ind >= numContacts) {
	    return;
	}

	dstCounts[ind] = (srcContact[ind].timeOfImpact < GJK_STEPSIZE && float4_length(srcContact[ind].separateAxis) > 1e-8f);	
}

__global__ void squeezeContactPairs_kernel(uint2 * dstPairs, uint2 * srcPairs,
                                    ContactData * dstContact, ContactData *srcContact,
									uint * counts, uint * packLocs, 
									uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(!counts[ind]) return;
	
	const uint toLoc = packLocs[ind];
	dstPairs[toLoc] = srcPairs[ind];
	dstContact[toLoc] = srcContact[ind];
}

__global__ void resetX_kernel(float3 * dst, float3 * src, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
    ind <<= 3;
    int i;
    for(i=0; i < 8; i++) {
        if(ind>= maxInd) return;
        
        dst[ind] = src[ind];
        ind++;
    }
}

extern "C" {

void narrowphaseComputeSeparateAxis(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart,
		uint numOverlappingPairs)
{
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    computeSeparateAxis_kernel<<< grid, block >>>(dstContact, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

void narrowphaseComputeTimeOfImpact(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs)
{   
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    computeTimeOfImpact_kernel<<< grid, block >>>(dstContact, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

void narrowphase_computeInitialSeparation(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs)
{   
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    computeInitialSeparation_kernel<<< grid, block >>>(dstContact, 
                                                        pairs, 
                                                        pos, 
                                                        vel, 
                                                        ind, 
                                                        pointStart, 
                                                        indexStart, 
                                                        numOverlappingPairs);
}

void narrowphase_advanceTimeOfImpactIterative(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs)
{   
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    advanceTimeOfImpactIterative_kernel<<< grid, block >>>(dstContact, 
                                                        pairs, 
                                                        pos, 
                                                        vel, 
                                                        ind, 
                                                        pointStart, 
                                                        indexStart, 
                                                        numOverlappingPairs);
}

void narrowphaseComputeValidPairs(uint * dstCounts, 
        ContactData * srcContact,
        uint numContacts, 
        uint scanBufferLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(scanBufferLength, 512);
    dim3 grid(nblk, 1, 1);
    
    computeValidPairs_kernel<<< grid, block >>>(dstCounts, srcContact, numContacts, scanBufferLength);
}

void narrowphaseSqueezeContactPairs(uint2 * dstPairs, uint2 * srcPairs,
                                    ContactData * dstContact, ContactData *srcContact,
									uint * counts, uint * packLocs, 
									uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    squeezeContactPairs_kernel<<< grid, block >>>(dstPairs, srcPairs,
                                    dstContact, srcContact,
									counts, packLocs, 
									maxInd);
}

void narrowphaseResetX(float3 * dst, float3 *src, uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd>>3, 512);
    dim3 grid(nblk, 1, 1);
    
    resetX_kernel<<< grid, block >>>(dst, src, maxInd);
}

}
