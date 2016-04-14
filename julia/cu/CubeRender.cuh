#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include <cu/VoxelPrim.cuh>
#include "cuSMem.cuh"

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
    box.low.x = 11.f;
    box.low.y = -15.f;
    box.low.z = -5.f;
    box.high.x = 21.f;
    box.high.y = -3.f;
    box.high.z = 15.f;
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = 1.f;
    incident.d.w = 1e28f;
    
    float tmin, tmax;
    if(ray_box(incident, box, tmin, tmax) ) {
        float3 hitP;
        ray_progress(hitP, incident, tmin - 1e-5f);
        float3 hitN = c_ray_box_face[side1_on_aabb4<float4>(box, hitP)];
	
    int r = 128 + 127 * hitN.x;
    int g = 128 + 127 * hitN.y;
    int b = 128 + 127 * hitN.z;
    
	pix[ind] = encodeRGB(r, g, b);
	}
}

__global__ void onePyrmaid_kernel(uint * pix, 
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
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = -1e28f;
    incident.d.w = 1e28f;
    
    float3 hitP, hitN;
    if(ray_box_and_hull(hitP, hitN, incident, box, planes, 5) ) {
    int r = 128 + 127 * hitN.x;
    int g = 128 + 127 * hitN.y;
    int b = 128 + 127 * hitN.z;
	pix[ind] = encodeRGB(r, g, b);
	}
	else {
	        pix[ind] = encodeRGB(99, 99, 99);
	}
}

__global__ void oneVoxel_kernel(uint * pix, 
                                float * depth,
                                Voxel * voxels  )
{
    float *sdata = SharedMemory<float>();
    Aabb4 * sboxes = (Aabb4 *)sdata; /// 8 * 64
/// max 64 boxes
/// not enough shared mem to store all contours for all boxes
    int * sncontours = (int *)&sdata[512]; /// 1 * 64
    int * scontours = (int *)&sdata[576]; /// 1 * 64
    float3 * scurrentContourN = (float3 *)&sdata[1152]; /// 16 * 3
    float * scurrentContourD = &sdata[1200]; /// 16 * 1
    int * scurrentActiveContour = (int *)&sdata[1216]; /// 16 * 1
    float3 * sboxN = (float3 *)&sdata[1232]; /// 3 * 6
    float * sboxD = &sdata[1250];
    
    int tidx = threadIdx.x + blockDim.x * threadIdx.y;
    if(tidx < 5) {
        Voxel v = voxels[tidx];
        sboxes[tidx] = calculate_bbox(v);
        int nct = get_n_contours(v);
        sncontours[tidx] = nct;
        for(int i=0; i< nct; ++i)
            scontours[tidx * 8 + i] = v.m_contour[i];
        
    }
    __syncthreads();
    
    uint px = getPixelCoordx();
    uint py = getPixelCoordy();
    
    Ray4 incident;
    
    v3_convert<float4, float3>(incident.o, c_frustumVec[0]);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[1], px);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[2], py);
    
    v3_convert<float4, float3>(incident.d, c_frustumVec[3]);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[4], px);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[5], py);
    
    v3_minus<float4, float4>(incident.d, incident.o);
    v3_normalize_inplace<float4>(incident.d);
    
    incident.o.w = -1e20f;
    incident.d.w = 1e20f;

    float t0, t1, mint0 = 1e20f, preMint0 = 1e20f;
    float3 t0Normal, t1Normal;
    float3 shadingN = make_float3(0.f, 0.f, 0.f);
     
    for(int i=0;i<5;++i) {
        
        if(tidx < sncontours[i]) {
            extractCurrentVoxelContour(scurrentContourN,
                                scurrentContourD,
                                scurrentActiveContour,
                                &scontours[i * 8],
                                sboxes[i],
                                tidx);
            
        }
        
        int ibox = tidx - sncontours[i];
        if(ibox >= 0 && ibox < 6) {
            extractBoxND(sboxN, sboxD,
                                sboxes[i],
                                ibox);
            
        }
        __syncthreads();
        
        if(ray_box_hull2(t0, t1, 
                        t0Normal, t1Normal,
                        incident, 
                        sboxN, sboxD) ) {

           if(t0 < mint0) {
                preMint0 = mint0;
                mint0 = t0;
                
                if(ray_voxel_hull2(t0, t1, 
                        t0Normal, t1Normal, 
                        incident,
                        scurrentContourN,
                        scurrentContourD,
                        scurrentActiveContour,
                        sncontours[i] * 2) ) {
                                  
                    shadingN = t0Normal;

                }
                else {
                    mint0 = preMint0;
                    
                }
            }
        }
        
        __syncthreads();
	}
	
	if(px < c_renderRect.x || px >= c_renderRect.z) return;
    if(py < c_renderRect.y || py >= c_renderRect.w) return;
    
/// output	       
    uint ind = getTiledPixelIdx();
    
	pix[ind] = encodeRGB(128 + 127 * shadingN.x, 
	                    128 + 127 * shadingN.y,
	                    128 + 127 * shadingN.z);
}

