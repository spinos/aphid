#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

__global__ void showTile_kernel(uint * pix, 
                                float * depth)
{
    uint ind = getTiledPixelIdx();
    
    int r = 255 * (float)blockIdx.x / (float)gridDim.x;
    int g = 255 * (float)blockIdx.y / (float)gridDim.y;
    
	pix[ind] = encodeRGB(r, g, 0);
}

__global__ void showRay_kernel(uint * pix, 
                                float * depth)
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
    
    uint ind = getTiledPixelIdx();
    
    int r = 128 + 128 * incident.d.x;
    int g = 128 + 128 * incident.d.y;
    
	pix[ind] = encodeRGB(r, g, 0);
}

__global__ void oneCube_kernel(uint * pix, 
                                float * depth)
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
    box.low.x = -5.3f;
    box.low.y = 8.f;
    box.low.z = -5.f;
    box.high.x = 5.f;
    box.high.y = 18.f;
    box.high.z = 5.2f;
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = 1.f;
    incident.d.w = 1e28f;
    
    float tmin, tmax;
    if(ray_box(incident, box, tmin, tmax) ) {
        float3 hitP;
        ray_progress(hitP, incident, tmin - 1e-4f);
        float3 hitN = c_ray_box_face[side_on_aabb4(box, hitP)];
	
    int r = 128 + 127 * hitN.x;
    int g = 128 + 127 * hitN.y;
    int b = 128 + 127 * hitN.z;
    
	pix[ind] = encodeRGB(r, g, b);
	}
}
