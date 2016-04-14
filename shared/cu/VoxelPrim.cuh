#ifndef VOXEL_PRIM_CUH
#define VOXEL_PRIM_CUH

#include "RayIntersection.cuh"

struct Voxel {
    int m_pos;
    int m_level;
    int m_color;
    int m_contour[7];		
};

inline __device__ uint Compact1By2(uint x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

inline __device__ void decodeMorton3D(uint code, unsigned & x, unsigned & y, unsigned & z)
{
    x = Compact1By2(code >> 2);
    y = Compact1By2(code >> 1);
    z = Compact1By2(code >> 0);
}

inline __device__ void decode_color12(float3 & c,
					const int & src)
{
	int masked = src & 4095;
	int red = (masked>>8) & 15;
	int green = (masked>>4) & 15;
	int blue = masked & 15;
	
/// to center
	c.x = (float)red * 0.0625f + 0.03125f;
	c.y = (float)green * 0.0625f + 0.03125f;
	c.z = (float)blue * 0.0625f + 0.03125f;
}

inline __device__ void decode_colnor30_nor(float3 & n, const int & src)
{
	int d = src & 32768;
	int axis = (src>>16) & 3;
	int u = (src>>18) & 63;
	int v = (src>>24) & 63;
	if(axis ==0) {
		n.y = (float)(u - 32) / 31.f;
		n.z = (float)(v - 32) / 31.f;
		n.x = 1.f;
		if(d==0) n.x = -1.f;
	}
	else if(axis == 1) {
		n.x = (float)(u - 32) / 31.f;
		n.z = (float)(v - 32) / 31.f;
		n.y = 1.f;
		if(d==0) n.y = -1.f;
	}
	else {
		n.x = (float)(u - 32) / 31.f;
		n.y = (float)(v - 32) / 31.f;
		n.z = 1.f;
		if(d==0) n.z = -1.f;
	}
	v3_normalize_inplace<float3>(n);
}

inline __device__ float3 get_contour_point(const int & x,
                                const float4 & o,
                                const float & d)
{
    float3 r;
	decode_color12(r, x);
	r.x = d * r.x + o.x;
	r.y = d * r.y + o.y;
	r.z = d * r.z + o.z;
	return r;
}

inline __device__ float3 get_contour_normal(const int & x)
{ 
    float3 r;
    decode_colnor30_nor(r, x);
    return r;
}

inline __device__ float get_contour_thickness(const int & x,
                                const float & d)
{
    int ng = (x & 32767)>>12;
    return (float)ng * d * .125f;
}


inline __device__ int get_n_contours(const Voxel & v)
{
    return (v.m_level & 255) >> 4;
}

inline __device__ Aabb4 calculate_bbox(const Voxel & v) 
{
    uint x, y, z;
	decodeMorton3D(v.m_pos, x, y, z);
	float h = 1<< (8 - (v.m_level & 15) );
	
	Aabb4 b;
	b.low.x = x - h;
	b.low.y = y - h;
	b.low.z = z - h;
	b.high.x = x + h;
	b.high.y = y + h;
	b.high.z = z + h;
	
	return b;
}

inline __device__ int ray_voxel_hull(float &t0, float & t1,
                        float3 & hitP, float3 & hitN,
                        Ray4 & ray,
                        const Voxel & v,
                        const Aabb4 & box)
{
    t0 = -1e20f;
    t1 = 1e20f;
    
    const float voxelSize = box.high.x - box.low.x;
    
    float t, denom, d, thickness;
    float3 t0Normal, t1Normal, pslab, pnt, nor;
    int nct = get_n_contours(v);
    int i=0;
    for(;i<nct;++i) {
        int contour = v.m_contour[i];
        pnt = get_contour_point(contour, box.low, voxelSize);
        nor = get_contour_normal(contour);
        d = -v3_dot<float3, float3>(nor, pnt);
        thickness = get_contour_thickness(contour, voxelSize * .866f);
        
        if(thickness > 0.f) {
/// slab
            pslab = pnt;
            v3_add_mult<float3, float3, float>(pslab, nor, thickness);
            d = -v3_dot<float3, float3>(nor, pslab);
            
            if(ray_plane(t, denom, ray, nor, d) )
                update_tnormal(t0, t1, t0Normal, t1Normal, nor, t, denom);
            
            if(t0 >= t1)
                return 0;
            
            pslab = pnt;
            v3_reverse_inplace<float3>(nor);
            v3_add_mult<float3, float3, float>(pslab, nor, thickness);
            d = -v3_dot<float3, float3>(nor, pslab);
            
            if(ray_plane(t, denom, ray, nor, d) )
                update_tnormal(t0, t1, t0Normal, t1Normal, nor, t, denom);
            
            if(t0 >= t1) 
                return 0;
            
        }
        else {
            
            if(ray_plane(t, denom, ray, nor, d) )
                update_tnormal(t0, t1, t0Normal, t1Normal, nor, t, denom);
        
            if(t0 >= t1)
                return 0;
            
        }
        if(t0 > ray.d.w) return 0;
    }
    
    hitN = t0Normal;
    ray_progress(hitP, ray, t0);
    return 1;
}

inline __device__ int ray_voxel(float3 & hitP, float3 & hitN,
                        Ray4 & ray,
                        const Voxel & v)
{
    Aabb4 box = calculate_bbox(v);
    
    float3 boxN, boxP;
    float boxt0;
    float boxt1;
    
    if(!ray_box_hull(boxt0, boxt1, boxP, boxN, ray, box) )
        return 0;
    
    if(boxt0 >= ray.d.w) return 0;
    
    hitP = boxP;
    hitN = boxN;
    
    ray.o.w = boxt0;
    ray.d.w = boxt1;
    
    float3 hullP, hullN;
    float hullt0, hullt1;
    
    if(ray_voxel_hull(hullt0, hullt1, hullP, hullN, ray, v, box) ) {
            
        if(hullt0 > boxt0) {
            hitP = hullP;
            hitN = hullN;
            ray.o.w = hullt0;
        }   
         
        if(hullt1 < boxt1)
            ray.d.w = hullt1;
    }
    else {
        ray.d.w = 1e20f;
        return 0;
    }

    
    return 1;
}


inline __device__ int ray_voxel_hull1(float &t0, float & t1,
                        float3 & t0Normal, float3 & t1Normal,
                        Ray4 & ray,
                        const Voxel & v,
                        const Aabb4 & box)
{
    t0 = -1e20f;
    t1 = 1e20f;
    
    const float voxelSize = box.high.x - box.low.x;
    
    float d, thickness;
    float3 pslab, pnt, nor;
    int nct = get_n_contours(v);
    int i=0;
    for(;i<nct;++i) {
        int contour = v.m_contour[i];
        pnt = get_contour_point(contour, box.low, voxelSize);
        nor = get_contour_normal(contour);
        d = -v3_dot<float3, float3>(nor, pnt);
        thickness = get_contour_thickness(contour, voxelSize * .866f);
        
        if(thickness > 0.f) {
/// slab
            pslab = pnt;
            v3_add_mult<float3, float3, float>(pslab, nor, thickness);
            d = -v3_dot<float3, float3>(nor, pslab);
            
            ray_plane1(t0, t1, t0Normal, t1Normal, ray, nor, d);
            if(t0 >= t1)
            return 0;
        
            pslab = pnt;
            v3_reverse_inplace<float3>(nor);
            v3_add_mult<float3, float3, float>(pslab, nor, thickness);
            d = -v3_dot<float3, float3>(nor, pslab);
            
            ray_plane1(t0, t1, t0Normal, t1Normal, ray, nor, d);
            if(t0 >= t1)
            return 0;
            
        }
        else {
            
            ray_plane1(t0, t1, t0Normal, t1Normal, ray, nor, d);
            if(t0 >= t1)
            return 0;
        }
        

    }
    
    if(t0 > ray.d.w) return 0;
    return 1;
}

inline __device__ int get_closest_voxel(
                    float & t0,
                    float & t1,
                    float3 & t0Normal, 
                    float3 & t1Normal,
                    Aabb4 & outBox,
                    Ray4 & incident,
                    Voxel * voxels, 
                    int n)
{
    float3 n0, n1;
    int r = -1;
    for(int i=0;i<n;++i) {
        Voxel v = voxels[i];
        Aabb4 box = calculate_bbox(v);
        if(ray_box_hull1(t0, t1, 
            n0, n1,
            incident, box) ) {
        
            
            t0Normal = n0;
            t1Normal = n1;
            incident.d.w = t1;
            r = i;
            outBox = box;
        
        }
    }
    return r;
}

#endif
