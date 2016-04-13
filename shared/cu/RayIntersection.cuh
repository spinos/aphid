#ifndef RAY_INTERSECTION_CUH
#define RAY_INTERSECTION_CUH

#include "VectorMath.cuh"

struct __align__(16) Ray4 {
	float4 o;	// origin
	float4 d;	// direction
};

struct __align__(4) Aabb {
    float3 low;
    float3 high;
};

struct __align__(16) Aabb4 {
    float4 low;
    float4 high;
};

/// center and extent
typedef float4 Cube;

__constant__ float3 c_ray_box_face[6];

inline __device__ void aabb4_from_cube(Aabb4 & v, const Cube & src)
{ 
    v.low.x = src.x - src.w; 
    v.low.y = src.y - src.w; 
    v.low.z = src.z - src.w; 
    v.high.x = src.x + src.w; 
    v.high.y = src.y + src.w; 
    v.high.z = src.z + src.w;
}

template<typename T>
inline __device__ void aabb4_convert(Aabb4 & v, const T & src)
{ 
    v3_convert<float4, float3>(v.low, src.low); 
    v3_convert<float4, float3>(v.high, src.high); 
}

inline __device__ void aabb4_r(Aabb4 & v, const float * src)
{ 
    v3_r<float4>(v.low, src); 
    v3_r<float4>(v.high, &src[3]); 
}

inline __device__ void aabb4_reset(Aabb4 & b)
{
    b.low.x = b.low.y = b.low.z = 1e20f;
    b.high.x = b.high.y = b.high.z = -1e20f;
}

template<typename T>
inline __device__ void aabb4_expand(Aabb4 & b, 
                                    const T & pnt)
{
    if(b.low.x > pnt.x) b.low.x = pnt.x;
    if(b.low.y > pnt.y) b.low.y = pnt.y;
    if(b.low.z > pnt.z) b.low.z = pnt.z;
    if(b.high.x < pnt.x) b.high.x = pnt.x;
    if(b.high.y < pnt.y) b.high.y = pnt.y;
    if(b.high.z < pnt.z) b.high.z = pnt.z;
}

inline __device__ void aabb4_split(Aabb4 & lftBox, Aabb4 & rgtBox, 
                                const Aabb4 & box, 
                                const int & axis, 
                                const float & splitPos)
{
    lftBox = box;
    rgtBox = box;
    if(axis == 0) {
        lftBox.high.x = splitPos;
        rgtBox.low.x = splitPos;
    }
    else if(axis == 1) {
        lftBox.high.y = splitPos;
        rgtBox.low.y = splitPos;
    }
    else {
        lftBox.high.z = splitPos;
        rgtBox.low.z = splitPos;
    }
}

inline __device__ int aabb4_touch(const Aabb4 & a,
                                const Aabb4 & b)
{
    if(a.low.x >= b.high.x || a.high.x <= b.low.x) return 0;
	if(a.low.y >= b.high.y || a.high.y <= b.low.y) return 0;
	if(a.low.z >= b.high.z || a.high.z <= b.low.z) return 0;
	return 1;
}

template<typename Tbox, typename Tpnt>
inline __device__ int is_v3_inside(const Tbox & box, 
                                    const Tpnt & p)
{
    if(p.x < box.low.x || p.x > box.high.x) return 0;
	if(p.y < box.low.y || p.y > box.high.y) return 0;
	if(p.z < box.low.z || p.z > box.high.z) return 0;
	return 1;
}

inline __device__ int is_approximate(const float & a, const float & b)
{ return absoluteValueF(a-b) < 1e-5f; }

inline __device__ int side_on_aabb4(const Aabb4 & b,
                                    const float3 & pnt)
{
    if(b.low.x - pnt.x  > 1e-5f ) return 0;
    if(pnt.x - b.high.x > 1e-5f ) return 1;
    if(b.low.y - pnt.y  > 1e-5f ) return 2;
    if(pnt.y - b.high.y > 1e-5f ) return 3;
    if(b.low.z - pnt.z  > 1e-5f ) return 4;
    return 5;    
}

inline __device__ int on_edge_aabb4(const Aabb4 & b,
                                    const float3 & pnt)
{
    float3 r;
    r.x = pnt.x - (b.low.x + b.high.x) * .5f;
    r.y = pnt.y - (b.low.y + b.high.y) * .5f;
    r.z = pnt.z - (b.low.z + b.high.z) * .5f;
    r.x /= b.high.x - b.low.x;
    r.y /= b.high.y - b.low.y;
    r.z /= b.high.z - b.low.z;
    v3_normalize_inplace<float3>(r);
    
    int jr = v3_major_axis<float3>(r);
	if(jr == 0) {
		if(r.x == r.y || r.x == r.z) return 1;
		return 0;
	}
	if(jr == 1) {
		if(r.y == r.x || r.y == r.z) return 1;
		return 0;
	}
	if(r.z == r.x || r.z == r.y) return 1;
	return 0;   
}

template<typename Td>
inline __device__ int side1_on_aabb4(const Aabb4 & b,
                                    const float3 & pnt)
{
    float3 r;
    r.x = pnt.x - (b.low.x + b.high.x) * .5f;
    r.y = pnt.y - (b.low.y + b.high.y) * .5f;
    r.z = pnt.z - (b.low.z + b.high.z) * .5f;
    r.x /= b.high.x - b.low.x;
    r.y /= b.high.y - b.low.y;
    r.z /= b.high.z - b.low.z;
    int jr = v3_major_axis<float3>(r);
	if(jr == 0) {
	    if(r.x < 0.f) return 0;
		return 1;
	}
	if(jr == 1) {
		if(r.y < 0.f) return 2;
		return 3;
	}
	if(r.z < 0.f) return 4;
	return 5;   
}

inline __device__ void ray_progress(float3 & p, const Ray4 & r, float h)
{ 
  p.x = r.o.x + r.d.x * h;
  p.y = r.o.y + r.d.y * h;
  p.z = r.o.z + r.d.z * h; 
}

inline __device__ void weightedSum(float3 & dst,
                            float3 & v1,
                            float3 & v2,
                            float3 & v3,
                            float w1,
                            float w2,
                            float w3)
{
    dst.x = v1.x * w1 + v2.x * w2 + v3.x * w3;
    dst.y = v1.y * w1 + v2.y * w2 + v3.y * w3;
    dst.z = v1.z * w1 + v2.z * w2 + v3.z * w3;
}

inline __device__ int ray_box(const Ray4 & ray,
                        const Aabb4 & aabb,
                        float & tmin, float & tmax)
{
	//AABB is considered as 3 pairs of 2 planes( {x_min, x_max}, {y_min, y_max}, {z_min, z_max} ).
	//t_min is the point of intersection with the closer plane, t_max is the point of intersection with the farther plane.
	//
	//if (rayNormalizedDirection.x < 0.0f), then max.x will be the near plane 
	//and min.x will be the far plane; otherwise, it is reversed.
	//
	//In order for there to be a collision, the t_min and t_max of each pair must overlap.
	//This can be tested for by selecting the highest t_min and lowest t_max and comparing them.
	
	//int3 isNegative = isless( rayNormalizedDirection, make_float3(0.0f, 0.0f, 0.0f) );	//isless(x,y) returns (x < y)
	int3 isNegative = make_int3(ray.d.x < 0.f, ray.d.y < 0.f, ray.d.z < 0.f);
	//When using vector types, the select() function checks the most signficant bit, 
	//but isless() sets the least significant bit.
	//isNegative <<= 31;

	//select(b, a, condition) == condition ? a : b
	//When using select() with vector types, (condition[i]) is true if its most significant bit is 1
	float3 t_min = float4_difference( select4(aabb.high, aabb.low, isNegative), ray.o );
	float3 t_max = float4_difference( select4(aabb.low, aabb.high, isNegative), ray.o );
	
	v3_divide_inplace<float3, float4>(t_min, ray.d);
	v3_divide_inplace<float3, float4>(t_max, ray.d);
	
	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	tmin = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, ray.o.w)) );
	tmax = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, ray.d.w)) );

	if(tmin >= tmax || tmax <= 0.f) return 0;
	
	return 1;
}

inline __device__ int ray_plane(float & t, float & denom,
                                const Ray4 & ray,
                                const float3 & n, 
                                const float & d)
{
    denom = v3_dot<float3, float4>(n, ray.d);
/// parallel
    if(absoluteValueF(denom) < 1e-8f) return 0;
    
    t = -(d + v3_dot<float3, float4>(n, ray.o) ) / denom;
    
    return 1;
}

inline __device__ void update_tnormal(float & t0, float & t1,
                        float3 & t0Normal, float3 & t1Normal,
                        const float3 & n,
                        float t, float denom)
{
    if(denom < 0.f) {
/// last enter
        if(t > t0) {
            t0 = t;
            t0Normal = n;
        }
    } else {
/// first exit
        if(t < t1) {
            t1 = t;
            t1Normal = n;
        }
    }
}

inline __device__ int ray_box_hull(float3 & hitP, float3 & hitN,
                        Ray4 & ray,
                        const Aabb4 & box)
{
    float t, denom;
    float3 t0Normal, t1Normal;
    float3 n;
    n.x = -1.f; n.y = 0.f; n.z = 0.f;
    float d = - n.x * box.low.x - n.y * box.low.y - n.z * box.low.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    n = make_float3(1.f, 0.f, 0.f);
    d = - n.x * box.high.x - n.y * box.low.y - n.z * box.low.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    n = make_float3(0.f, -1.f, 0.f);
    d = - n.x * box.low.x - n.y * box.low.y - n.z * box.low.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    n = make_float3(0.f, 1.f, 0.f);
    d = - n.x * box.low.x - n.y * box.high.y - n.z * box.low.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    n = make_float3(0.f, 0.f, -1.f);
    d = - n.x * box.low.x - n.y * box.low.y - n.z * box.low.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    n = make_float3(0.f, 0.f, 1.f);
    d = - n.x * box.low.x - n.y * box.low.y - n.z * box.high.z;
    if(ray_plane(t, denom, ray, n, d))
        update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
    
    if(ray.o.w > ray.d.w) return 0;
    
    hitN = t0Normal;
    ray_progress(hitP, ray, ray.o.w);
    return 1;
}

inline __device__ int ray_box_and_hull(float3 & hitP, float3 & hitN,
                        Ray4 & ray,
                        const Aabb4 & box,
                        float4 * planes,
                        int numPlanes)
{
    if(!ray_box_hull(hitP, hitN, ray, box) )
        return 0;
        
    float tEnterBox = ray.o.w;
    float3 n, t0Normal, t1Normal;
    float t, denom, d;
    int i=0;
    for(;i<numPlanes;++i) {
        float4 plane = planes[i];
        v3_convert<float3, float4>(n, plane);
        d = plane.w;
        if(ray_plane(t, denom, ray, n, d) )
            update_tnormal(ray.o.w, ray.d.w, t0Normal, t1Normal, n, t, denom);
        
        if(ray.o.w > ray.d.w)
            return 0;
        
    }
    
    if(ray.o.w > tEnterBox) {
        ray_progress(hitP, ray, ray.o.w);
        hitN = t0Normal;
    }
        
    return 1;
}
#endif        //  #ifndef RAY_INTERSECTION_CUH

