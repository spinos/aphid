#ifndef _GJK_MATH_CU_
#define _GJK_MATH_CU_

#include "bvh_common.h"
#include "bvh_math.cuh"
#include "barycentric.cu"
#include "line_math.cu"
#include "triangle_math.cu"

#define GJK_MAX_NUM_ITERATIONS 10
#define GJK_THIN_MARGIN 0.04f
#define GJK_THIN_MARGIN2 0.08f
#define GJK_STEPSIZE 0.01666667f
#define FLOAT3_ORIGIN make_float3(0.f, 0.f, 0.f)

struct Simplex {
    float3 p[4];
	float3 pA[4];
	float3 pB[4];
	int dimension;
};

struct TetrahedronProxy {
    float3 p[4];
};

inline __device__ void resetSimplex(Simplex & s)
{ s.dimension = 0; }

inline __device__ void addToSimplex(Simplex & s, const float3 & p)
{
    if(s.dimension == 0) {
        s.p[0] = p;
        s.pA[0] = s.pA[3];
        s.pB[0] = s.pB[3];
        s.dimension = 1;
    }
    else if(s.dimension == 1) {
		if(distance2_between(p, s.p[0]) > 1e-6f) {
		    s.p[1] = p;
		    s.pA[1] = s.pA[3];
		    s.pB[1] = s.pB[3];
		    s.dimension = 2;
		}
    }
    else if(s.dimension == 2) {
		if(!isTriangleDegenerate(s.p[0], s.p[1], p)) {
		    s.p[2] = p;
		    s.pA[2] = s.pA[3];
		    s.pB[2] = s.pB[3];
		    s.dimension = 3;
		}
    }
    else {
        if(!isTetrahedronDegenerate(s.p[0], s.p[1], s.p[2], p)) {
		    s.p[3] = p;
		    // s.pA[3] = localA;
		    // s.pB[3] = localB;
		    s.dimension = 4;
		}
    }
}

inline __device__ void addToSimplex(Simplex & s, const float3 & p, const float3 & localA, const float3 & localB)
{
    if(s.dimension < 1) {
        s.p[0] = p;
        s.pA[0] = localA;
        s.pB[0] = localB;
        s.dimension = 1;
    }
    else if(s.dimension < 2) {
		if(distance2_between(p, s.p[0]) > 1e-6f) {
		    s.p[1] = p;
		    s.pA[1] = localA;
		    s.pB[1] = localB;
		    s.dimension = 2;
		}
    }
    else if(s.dimension < 3) {
		if(!isTriangleDegenerate(s.p[0], s.p[1], p)) {
		    s.p[2] = p;
		    s.pA[2] = localA;
		    s.pB[2] = localB;
		    s.dimension = 3;
		}
    }
    else {
        if(!isTetrahedronDegenerate(s.p[0], s.p[1], s.p[2], p)) {
		    s.p[3] = p;
		    s.pA[3] = localA;
		    s.pB[3] = localB;
		    s.dimension = 4;
		}
    }
}

inline __device__ float3 initialPoint(const TetrahedronProxy & tet)
{
    if(float3_length2(tet.p[0]) > 1e-6f)
        return tet.p[0];
    
    return tet.p[1];
}

inline __device__ float3 initialPoint2(const TetrahedronProxy & tet, const float3 & ref)
{
    float3 r = float3_difference(tet.p[0], ref);
    
    if(float3_length2(r) < 1e-6f)
        r = float3_difference(tet.p[1], ref);
    
    return r;
}

inline __device__ float3 supportPoint(TetrahedronProxy tet, float3 ref, float margin, float3 & localP)
{
    float maxDotv = -1e8f;
    float dotv;
    
    float3 dMargin = scale_float3_by(float3_normalize(ref), margin);
    float3 res, wp;
    
    int i;
    
    float3 center = tet.p[0];
    for(i=1; i<4; i++) {
        center = float3_add(center, tet.p[i]);
    }
    center = scale_float3_by(center,0.25f);
    
    for(i=0; i<4; i++) {
        wp = float3_add(tet.p[i], dMargin);
        dotv = float3_dot(wp, ref);
        if(dotv > maxDotv) {
            maxDotv = dotv;
            res = wp;
            localP = float3_difference(tet.p[i], center);
        }
    }
    
    return res;
}

inline __device__ int isOriginInsideSimplex(const Simplex & s)
{
    if(s.dimension > 3) {
        return pointInsideTetrahedronTest(FLOAT3_ORIGIN, s.p);
    }
    return 0;
}

inline __device__ int isPointInsideSimplex(const Simplex & s, const float3 & p)
{
    if(s.dimension > 3) {
        return pointInsideTetrahedronTest(p, s.p);
    }
    return 0;
}

inline __device__ void computeClosestPointOnSimplex(Simplex & s, const float3 & p, ClosestPointTestContext & ctc)
{
    ctc.closestDistance = 1e10f;

    if(s.dimension < 2) {
        ctc.closestPoint = s.p[0];
        ctc.closestDistance = distance_between(p, s.p[0]);
    }
    else if(s.dimension < 3) {
        computeClosestPointOnLine1(p, s.p[0], s.p[1], ctc);
    }
    else if(s.dimension < 4) {
        computeClosestPointOnTriangle2(p, s.p[0], s.p[1], s.p[2], ctc);
    }
    else {
        computeClosestPointOnTetrahedron(p, s.p, ctc);
    }
}

inline __device__ void computeClosestPointOnSimplex(Simplex & s, ClosestPointTestContext & ctc)
{
    computeClosestPointOnSimplex(s, FLOAT3_ORIGIN, ctc);
}

inline __device__ void computeContributionSimplex(BarycentricCoordinate & dst, const Simplex & s, const float3 & q)
{
    if(s.dimension < 2) {
        dst = make_float4(1.f, -1.f, -1.f, -1.f);
    }
    else if(s.dimension < 3) {
        dst = getBarycentricCoordinate2(q, s.p);
    }
    else if(s.dimension < 4) {
        dst = getBarycentricCoordinate3(q, s.p);
    }
    else {
        dst = getBarycentricCoordinate4(q, s.p);
    }
}

inline __device__ void interpolatePointAB(Simplex & s,
                                            const BarycentricCoordinate & contributes, 
                                            float3 & pA, float3 & pB)
{
	float3 qA = make_float3(0.f, 0.f, 0.f);
	float3 qB = make_float3(0.f, 0.f, 0.f);
	const float * wei = &contributes.x;
	int i;
	for(i =0; i < s.dimension; i++) {
		if(wei[i] > 1e-5f) {
		    float3_add_inplace(qA, scale_float3_by(s.pA[i], wei[i]));
		    float3_add_inplace(qB, scale_float3_by(s.pB[i], wei[i]));
		}
	}
	pA = qA;
	pB = qB;
}

inline __device__ void compareAndSwap(float * key, float3 * v1, float3* v2, float3 * v3, int a, int b)
{
    if(key[a] < key[b]) {
        float ck = key[a];
        key[a] = key[b];
        key[b] = ck;
        
        float3 cv = v1[a];
        v1[a] = v1[b];
        v1[b] = cv;
        
        cv = v2[a];
        v2[a] = v2[b];
        v2[b] = cv;
        
        cv = v3[a];
        v3[a] = v3[b];
        v3[b] = cv;
    }
}

inline __device__ void smallestSimplex(Simplex & s, BarycentricCoordinate & contributes)
{
	if(s.dimension < 2) return;
	
	float * bar = &contributes.x;
	
	compareAndSwap(bar, s.p, s.pA, s.pB, 0, 2);
	compareAndSwap(bar, s.p, s.pA, s.pB, 1, 3);
	compareAndSwap(bar, s.p, s.pA, s.pB, 0, 1);
	compareAndSwap(bar, s.p, s.pA, s.pB, 2, 3);
	compareAndSwap(bar, s.p, s.pA, s.pB, 1, 2);

	s.dimension = 0;
	
	if(bar[0] > 1e-5f) s.dimension++;
	if(bar[1] > 1e-5f) s.dimension++;
	if(bar[2] > 1e-5f) s.dimension++;
	if(bar[3] > 1e-5f) s.dimension++;
}

inline __device__ void computeSeparateDistance(Simplex & s, 
                                               const TetrahedronProxy & prxA,
                                               const TetrahedronProxy & prxB,
                                               const float & margin,
                                               ClosestPointTestContext & ctc,
                                               float4 & separateAxis,
                                               BarycentricCoordinate & coord)
{
    resetSimplex(s);

	float3 v = initialPoint(prxA);
	
	float3 w;
	float v2;
	int i = 0;
	
	while(i<GJK_MAX_NUM_ITERATIONS) {
	    w = float3_difference(supportPoint(prxA, float3_reverse(v), margin, s.pA[3]), 
	                            supportPoint(prxB, v, margin, s.pB[3]));
	    
	    v2 = float3_length2(v);
	    if((v2 - float3_dot(w, v)) < 0.0001f * v2) {
	        return;
	    }
	    
	    addToSimplex(s, w);
	    
	    if(isOriginInsideSimplex(s)) {
	        separateAxis.w = 0.f;
	        return;
	    }
	    
	    computeClosestPointOnSimplex(s, ctc);
	    
	    v = ctc.closestPoint;
	    separateAxis = make_float4(v.x, v.y, v.z, 1.f);
	    
	    computeContributionSimplex(coord, s, ctc.closestPoint);
	    
	    smallestSimplex(s, coord);
	    
	    i++;
	}
}

inline __device__ void checkClosestDistance(Simplex & s, 
                                        const TetrahedronProxy & prxA,
                                        const TetrahedronProxy & prxB,
                                        ClosestPointTestContext & result,
                                        float4 & dstSA,
                                        float3 & dstPA,
                                        float3 & dstPB,
                                        BarycentricCoordinate & coord)
{
    float3 cenA = prxA.p[0];
	cenA = float3_add(cenA, prxA.p[1]);
	cenA = float3_add(cenA, prxA.p[2]);
	cenA = float3_add(cenA, prxA.p[3]);
	cenA = scale_float3_by(cenA, 0.25f);
	
	float3 cenB = prxB.p[0];
	cenB = float3_add(cenB, prxB.p[1]);
	cenB = float3_add(cenB, prxB.p[2]);
	cenB = float3_add(cenB, prxB.p[3]);
	cenB = scale_float3_by(cenB, 0.25f);
	
    resetSimplex(s);
	
	float3 la, lb;
	la = float3_difference(prxA.p[0], cenA);
	lb = float3_difference(prxB.p[0], cenB);
	addToSimplex(s, prxA.p[0], la, lb);
	la = float3_difference(prxA.p[1], cenA);
	lb = float3_difference(prxB.p[1], cenB);
	addToSimplex(s, prxA.p[1], la, lb);
	la = float3_difference(prxA.p[2], cenA);
	lb = float3_difference(prxB.p[2], cenB);
	// addToSimplex(s, prxA.p[2], la, lb);
	la = float3_difference(prxA.p[3], cenA);
	lb = float3_difference(prxB.p[3], cenB);
	addToSimplex(s, prxA.p[3], la, lb);

	computeClosestPointOnSimplex(s, cenB, result);
	
	float3 d = float3_difference(result.closestPoint, cenB);
	dstSA = make_float4(d.x, d.y, d.z, 1.f);
	
	computeContributionSimplex(coord, s, result.closestPoint);
	
	// interpolatePointAB(s, coord, dstPA, dstPB);
}

#endif        //  #ifndef _GJK_MATH_CU_

