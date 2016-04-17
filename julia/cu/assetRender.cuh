#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include <cu/VoxelPrim.cuh>
#include "cuSMem.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

__global__ void assetPyrmaid_kernel(uint * pix, 
                                float * depth,
                                float4 * planes,
                                Aabb * bounding   )
{
    uint px = getPixelCoordx();
    uint py = getPixelCoordy();
    
    if(px < c_renderRect.x || px >= c_renderRect.z) return;
    if(py < c_renderRect.y || py >= c_renderRect.w) return;
    
    Ray4 incident;
    
    v3_convert<float4, float3>(incident.o, c_frustumVec[0]);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[1], px);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[2], py);
    
    v3_convert<float4, float3>(incident.d, c_frustumVec[3]);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[4], px);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[5], py);
    
    v3_minus<float4, float4>(incident.d, incident.o);
    v3_normalize_inplace<float4>(incident.d);
    
    Aabb4 box;
    aabb4_convert<Aabb>(box, *bounding);
    
    uint ind = getImagePixelIdx(px, py);
    
    incident.o.w = -1e28f;
    incident.d.w = 1e28f;
    
    float t0, t1;
    float3 t0Normal, t1Normal;
    float3 shadingNormal = make_float3(0.f, 0.f, 0.f);
    if(ray_box_slab(t0, t1,
                    t0Normal, t1Normal, 
                    incident, box) ) {
        
        if(ray_hull(t0, t1, 
                    t0Normal, t1Normal, 
                    incident,
                    planes, 5) ) {
            shadingNormal = t0Normal;
        }
	}
	
	pix[ind] = encodeRGB(128 + 127 * shadingNormal.x,
                            128 + 127 * shadingNormal.y,
                            128 + 127 * shadingNormal.z);
	
}


