#ifndef BARYCENTRIC_CU
#define BARYCENTRIC_CU

#include "bvh_math.cu"
#include "matrix_math.cu"

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

inline __device__ int isTriangleDegenerate(const float3 & a, const float3 & b, const float3 & c)
{
    float3 ab = float3_difference(b, a);
    float3 ac = float3_difference(c, a);
    float3 nor = float3_cross(ab, ac);
    return (float3_length2(nor) < 1e-6);
}

inline __device__ int isTetrahedronDegenerate(const float3 & a, const float3 & b, const float3 & c, const float3 & d)
{
    mat44 m;
    fill_mat44(m, a, b, c, d);
    
    float D0 = determinant44(m) ;
    if(D0 < 0.f) D0 = -D0;
    return (D0 < 1e-5);
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
	
    float D0 = float3_length2(n);
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

inline __device__ BarycentricCoordinate getBarycentricCoordinate4i(const float3 & p, const float3 * v, uint4 ind)
{
    BarycentricCoordinate coord;
    
    mat44 m;
    fill_mat44(m, v[ind.x], v[ind.y], v[ind.z], v[ind.w]);
    
    float D0 = determinant44(m);
    if(D0 == 0.f) {
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
    fill_mat44(m, p, v[ind.y], v[ind.z], v[ind.w]);
    coord.x = determinant44(m) / D0;
    fill_mat44(m, v[ind.x], p, v[ind.z], v[ind.w]);
    coord.y = determinant44(m) / D0;
    fill_mat44(m, v[ind.x], v[ind.y], p, v[ind.w]);
    coord.z = determinant44(m) / D0;
    fill_mat44(m, v[ind.x], v[ind.y], v[ind.z], p);
    coord.w = determinant44(m) / D0;
    
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

inline __device__ BarycentricCoordinate getBarycentricCoordinate4Relativei(const float3 & p, float3 * v, const uint4 & t)
{
    float3 q = v[t.x];
    q = float3_add(q, v[t.y]);
    q = float3_add(q, v[t.z]);
    q = float3_add(q, v[t.w]);
    q = scale_float3_by(q, .25f);
    q = float3_add(q, p);
    
    return getBarycentricCoordinate4i(q, v, t);
}

inline __device__ int pointInsideTriangleTest(const float3 & p, const float3 & nor, const float3 * tri)
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

inline __device__ int pointInsideTetrahedronTest(const float3 & p, const float3 * tet)
{
    BarycentricCoordinate coord = getBarycentricCoordinate4(p, tet);
    return (coord.x >=0 && coord.x <=1 && 
        coord.y >=0 && coord.y <=1 &&
        coord.z >=0 && coord.z <=1 &&
        coord.w >=0 && coord.w <=1);
}

inline __device__ void interpolate_float3i(float3 & dst, 
                                    uint4 ia,
                                    float3 * velocity,
                                    BarycentricCoordinate * coord)
{
    float3_set_zero(dst);
	
	dst = float3_add(dst, scale_float3_by(velocity[ia.x], coord->x));
	dst = float3_add(dst, scale_float3_by(velocity[ia.y], coord->y));
	dst = float3_add(dst, scale_float3_by(velocity[ia.z], coord->z));
	dst = float3_add(dst, scale_float3_by(velocity[ia.w], coord->w));
}
#endif        //  #ifndef BARYCENTRIC_CU

