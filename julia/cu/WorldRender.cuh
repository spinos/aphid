#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

__global__ void twoCube_kernel(uint * pix, 
                                float * nearDepth,
                                float * farDepth)
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
    box.low.x = -5.f;
    box.low.y = 8.f;
    box.low.z = -5.f;
    box.high.x = 5.f;
    box.high.y = 18.f;
    box.high.z = 5.f;
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = nearDepth[ind];
    incident.d.w = farDepth[ind];
    
    float3 hitP, hitN;
    if(ray_box(incident, hitP, hitN, box) ) {
    
    int r = 128 + 127 * hitN.x;
    int g = 128 + 127 * hitN.y;
    int b = 128 + 127 * hitN.z;
    
	pix[ind] = encodeRGB(r, g, b);
	}
}

