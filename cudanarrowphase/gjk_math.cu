#ifndef _GJK_MATH_H_
#define _GJK_MATH_H_

#include "bvh_math.cu"
#include "matrix_math.cu"

typedef float4 BarycentricCoordinate;

struct ClosestPointTestContext {
    float3 closestPoint;
    float closestDistance;
};

struct Simplex {
    float3 p[4];
	float3 pA[4];
	float3 pB[4];
	int dimension;
};

struct MovingTetrahedron {
    float3 p[4];
    float3 v[4];
};

struct TetrahedronProxy {
    float3 p[4];
};

inline __device__ float3 triangleNormal(const float3 * v)
{
    float3 ab = float3_difference(v[1], v[0]);
    float3 ac = float3_difference(v[2], v[0]);
    float3 nor = float3_cross(ab, ac);
    return float3_normalize(nor);
}

inline __device__ float3 triangleNormal2(const float3 * v)
{
    float3 ab = float3_difference(v[1], v[0]);
    float3 ac = float3_difference(v[2], v[0]);
    return float3_cross(ab, ac);
}

inline __device__ int isTriangleDegenerate(float3 a, float3 b, float3 c)
{
    float3 ab = float3_difference(b, a);
    float3 ac = float3_difference(c, a);
    float3 nor = float3_cross(ab, ac);
    return (float3_length2(nor) < 1e-6);
}

inline __device__ int isTetrahedronDegenerate(float3 a, float3 b, float3 c, float3 d)
{
    mat44 m;
    fill_mat44(m, a, b, c, d);
    
    float D0 = determinant44(m) ;
    if(D0 < 0.f) D0 = -D0;
    return (D0 < 0.01f);
}

inline __device__ void resetSimplex(Simplex & s)
{ s.dimension = 0; }

inline __device__ void addToSimplex(Simplex & s, float3 p, float3 localA, float3 localB)
{
    if(s.dimension < 1) {
        s.p[0] = p;
        s.pA[0] = localA;
        s.pB[0] = localB;
        s.dimension = 1;
    }
    else if(s.dimension < 2) {
		if(distance2_between(p, s.p[0]) > 1e-6) {
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

inline __device__ float3 initialPoint(const TetrahedronProxy & tet, const float3 & ref)
{
    float3 r = float3_difference(tet.p[0], ref);
    
    if(float3_length2(r) < 1e-6)
        r = float3_difference(tet.p[1], ref);
    
    return r;
}

inline __device__ float3 supportPoint(TetrahedronProxy tet, float3 ref, float margin, float3 & localP)
{
    float maxDotv = -1e8;
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

inline __device__ BarycentricCoordinate getBarycentricCoordinate2(const float3 & p, const float3 * v)
{
    BarycentricCoordinate coord;
	float3 dv = float3_difference(v[1], v[0]);
	float D0 = float3_length(dv);
	if(D0 < 1e-6) {
        coord.x = coord.y = coord.z = coord.w = -1.f;
    } 
    else {
        float3 dp = float3_difference(p, v[1]);
        coord.x = float3_length(dp) / D0;
        coord.y = 1.f - coord.x;
        coord.z = coord.w = -1.f;
	}
	return coord;
}


inline __device__ BarycentricCoordinate getBarycentricCoordinate3(const float3 & p, const float3 * v)
{
    BarycentricCoordinate coord;
    
    float3 n = triangleNormal2(v);
	
    float D0 = float3_length(n);
    if(D0 < 1e-6) {
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
	float3 na = float3_cross( float3_difference(v[2], v[1]), float3_difference(p, v[1]) );
    coord.x = float3_dot(n, na) / D0;
	float3 nb = float3_cross( float3_difference(v[0], v[2]), float3_difference(p, v[2]) );  
    coord.y = float3_dot(n, nb) / D0;
	float3 nc = float3_cross( float3_difference(v[1], v[0]), float3_difference(p, v[0]) );
    coord.z = float3_dot(n, nc) / D0;
    coord.w = -1.f;
    
    return coord;
}

inline __device__ BarycentricCoordinate getBarycentricCoordinate4(const float3 & p, const float3 * v)
{
    BarycentricCoordinate coord;
    
    mat44 m;
    fill_mat44(m, v[0], v[1], v[2], v[3]);
    
    float D0 = determinant44(m);
    if(D0 == 0.f) {
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
    fill_mat44(m, p, v[1], v[2], v[3]);
    coord.x = determinant44(m) / D0;
    fill_mat44(m, v[0], p, v[2], v[3]);
    coord.y = determinant44(m) / D0;
    fill_mat44(m, v[0], v[1], p, v[3]);
    coord.z = determinant44(m) / D0;
    fill_mat44(m, v[0], v[1], v[2], p);
    coord.w = determinant44(m) / D0;
    
    return coord;
}

inline __device__ int pointInsideTriangleTest(float3 p, float3 nor, float3 * tri)
{
    float3 e01 = float3_difference(tri[1], tri[0]);
	float3 x0 = float3_difference(p, tri[0]);
	if(float3_dot( float3_cross(e01, x0), nor ) < 0.f) return 0;
	
	float3 e12 = float3_difference(tri[2], tri[1]);
	float3 x1 = float3_difference(p, tri[1]);
	if(float3_dot( float3_cross(e12, x1), nor ) < 0.f) return 0;
	
	float3 e20 = float3_difference(tri[0], tri[2]);
	float3 x2 = float3_difference(p, tri[2]);
	if(float3_dot( float3_cross(e20, x2), nor ) < 0.f) return 0;
	
	return 1;
}

inline __device__ int pointInsideTetrahedronTest(float3 p, float3 * tet)
{
    BarycentricCoordinate coord = getBarycentricCoordinate4(p, tet);
    return (coord.x >=0 && coord.x <=1 && 
        coord.y >=0 && coord.y <=1 &&
        coord.z >=0 && coord.z <=1 &&
        coord.w >=0 && coord.w <=1);
}

inline __device__ int isPointInsideSimplex(Simplex & s, float3 p)
{
    if(s.dimension > 3) {
        return pointInsideTetrahedronTest(p, s.p);
    }
    return 0;
}

inline __device__ void computeClosestPointOnLine(float3 p, float3 * v, ClosestPointTestContext & result)
{
    float3 vr = float3_difference(p, v[0]);
    float3 v1 = float3_difference(v[1], v[0]);
	float dr = float3_length(vr);
	if(dr < 1e-6) {
        result.closestPoint = v[0];
		result.closestDistance = 0.f;
        return;
    }
	
	float d1 = float3_length(v1);
	vr = float3_normalize(vr);
	v1 = float3_normalize(v1);
	float vrdv1 = float3_dot(vr, v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = float3_add(v[0], scale_float3_by(v1, vrdv1));
	float dc = distance_between(v1, p);
	
	if(dc < result.closestDistance) {
	    result.closestPoint = v1;
	    result.closestDistance = dc;
	}
}

// http://mathworld.wolfram.com/Point-PlaneDistance.html

inline __device__ float3 projectPointOnPlane(float3 p, float3 v, float3 nor)
{
    float t = float3_dot(nor, v) - float3_dot(nor, p);
    return float3_add(p, scale_float3_by(nor, t));
}

inline __device__ void computeClosestPointOnTriangle(float3 p, float3 * v, ClosestPointTestContext & result)
{
    float3 nor = triangleNormal(v);
    float3 onplane = projectPointOnPlane(p, v[0], nor);
    
    if(pointInsideTriangleTest(onplane, nor, v)) {
        float d = distance_between(p, onplane);
        if(d < result.closestDistance) {
            result.closestPoint = onplane;
            result.closestDistance = d;
        }
        return;
    }
    
    computeClosestPointOnLine(p, v, result);
    float3 line[2];
    line[0] = v[1];
    line[1] = v[2];
    computeClosestPointOnLine(p, line, result);
    line[0] = v[2];
    line[1] = v[0];
    computeClosestPointOnLine(p, line, result);
}

inline __device__ void computeClosestPointOnTetrahedron(float3 p, float3 * v, ClosestPointTestContext & result)
{
	computeClosestPointOnTriangle(p, v, result);
	
	float3 pr[3];
	pr[0] = v[0];
	pr[1] = v[1];
	pr[2] = v[3];
	computeClosestPointOnTriangle(p, pr, result);
	
	pr[0] = v[0];
	pr[1] = v[2];
	pr[2] = v[3];
	computeClosestPointOnTriangle(p, pr, result);
	
	pr[0] = v[1];
	pr[1] = v[2];
	pr[2] = v[3];
	computeClosestPointOnTriangle(p, pr, result);
	
}

inline __device__ void computeClosestPointOnSimplex(Simplex & s, float3 p, ClosestPointTestContext & ctc)
{
    ctc.closestDistance = 1e10;

    if(s.dimension < 2) {
        ctc.closestPoint = s.p[0];
        ctc.closestDistance = distance_between(p, s.p[0]);
    }
    else if(s.dimension < 3) {
        computeClosestPointOnLine(p, s.p, ctc);
    }
    else if(s.dimension < 4) {
        computeClosestPointOnTriangle(p, s.p, ctc);
    }
    else {
        computeClosestPointOnTetrahedron(p, s.p, ctc);
    }
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
	pA = make_float3(0.f, 0.f, 0.f);
	pB = make_float3(0.f, 0.f, 0.f);
	/*
	if(contributes.x > 0.f) {
	    pA = float3_add(pA, scale_float3_by(s.pA[0], contributes.x));
	    pB = float3_add(pB, scale_float3_by(s.pB[0], contributes.x));
	}
	
	if(contributes.y > 0.f) {
	    pA = float3_add(pA, scale_float3_by(s.pA[1], contributes.y));
	    pB = float3_add(pB, scale_float3_by(s.pB[1], contributes.y));
	}
	
	if(contributes.z > 0.f) {
	    pA = float3_add(pA, scale_float3_by(s.pA[2], contributes.z));
	    pB = float3_add(pB, scale_float3_by(s.pB[2], contributes.z));
	}
	
	if(contributes.w > 0.f) {
	    pA = float3_add(pA, scale_float3_by(s.pA[3], contributes.w));
	    pB = float3_add(pB, scale_float3_by(s.pB[3], contributes.w));
	}
	*/
	const float * wei = &contributes.x;
	int i;
	for(i =0; i < s.dimension; i++) {
		if(wei[i] > 1e-5) {
		    pA = float3_add(pA, scale_float3_by(s.pA[i], wei[i]));
			pB = float3_add(pB, scale_float3_by(s.pB[i], wei[i]));
		}
	}
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
	
	if(bar[0] > 1e-5) s.dimension++;
	if(bar[1] > 1e-5) s.dimension++;
	if(bar[2] > 1e-5) s.dimension++;
	if(bar[3] > 1e-5) s.dimension++;
	
	/*
	int i, j;
	for(i = 0; i < s.dimension; i++) {
		if(bar[i] < 1e-5) {
			for(j = i; j < s.dimension - 1; j++) {
				s.p[j] = s.p[j+1];
				s.pA[j] = s.pA[j+1];
				s.pB[j] = s.pB[j+1];
				bar[j] = bar[j+1];
			}
			i--;
			s.dimension--;
		}
    }*/
}

inline __device__ void computeSeparateDistance(Simplex & s, float3 Pref, 
                                               const TetrahedronProxy & prxA,
                                               const TetrahedronProxy & prxB,
                                               ClosestPointTestContext & ctc,
                                               float4 & separateAxis,
                                               float3 & pointA,
                                               float3 & pointB,
                                               BarycentricCoordinate & coord)
{
	float3 v = initialPoint(prxA, Pref);
	
	float3 w, supportA, supportB, localA, localB;
	float margin = 0.02f;
	float v2;
	int i = 0;
	
	while(i<99) {
	    supportA = supportPoint(prxA, float3_reverse(v), margin, localA);
	    supportB = supportPoint(prxB, v, margin, localB);
	    
	    w = float3_difference(supportA, supportB);
	    
	    v2 = float3_length2(v);
	    if((v2 - float3_dot(w, v)) < 0.0001f * v2) {
	        break;
	    }
	    
	    addToSimplex(s, w, localA, localB);
	    
	    if(isPointInsideSimplex(s, Pref)) {
	        separateAxis.w = 0.f;
	        break;
	    }
	    
	    computeClosestPointOnSimplex(s, Pref, ctc);
	    
	    v = float3_difference(ctc.closestPoint, Pref);
	    separateAxis = make_float4(v.x, v.y, v.z, 1.f);
	    
	    computeContributionSimplex(coord, s, ctc.closestPoint);
	    
	    interpolatePointAB(s, coord, pointA, pointB);
		
	    smallestSimplex(s, coord);
	    
	    i++;
	}
}

inline __device__ void checkClosestDistance(Simplex & s, 
                                        const TetrahedronProxy & prxA,
                                        const TetrahedronProxy & prxB,
                                        ClosestPointTestContext & result)
{
    resetSimplex(s);
	
	float3 la, lb;
	addToSimplex(s, prxB.p[0], la, lb);
	addToSimplex(s, prxB.p[1], la, lb);
	addToSimplex(s, prxB.p[2], la, lb);
	addToSimplex(s, prxB.p[3], la, lb);
	// computeClosestPointOnSimplex(s, float3_add(prxA.p[2], scale_float3_by(prxA.v[2], 0.01667f)), result);
}

#endif        //  #ifndef _GJK_MATH_H_

