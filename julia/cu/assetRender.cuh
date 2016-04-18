#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include "cu/NTreeTraverse.cuh"
#include <cu/VoxelPrim.cuh>
#include "cuSMem.cuh"
#include "cuReduceInBlock.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

template<int NumThreads>
__global__ void assetBox_kernel(uint * pix, 
                                float * nearDepth,
                                float * farDepth,
                                NTreeBranch4 * branches,
                                NTreeLeaf * leaves,
                                Rope * ropes,
                                int * indirections,
                                Aabb4 * primitives)
{
    float *sdata = SharedMemory<float>();
/// 0  -> 7     grid translation and scaling
    float *sgts = &sdata[0];
/// 8  -> 15    grid tight box
    float *sgtb = &sdata[8];
/// 16 -> 23    grid box
    float *sgb = &sdata[16];
/// 24 - NumThreads + 23 
    int * srayActive = (int *)&sdata[24];

    const int tidx = threadIdx.x + blockDim.x * threadIdx.y;
    if(tidx < 1) {
        const NTreeBranch4 & rootBranch = branches[0];
/// get relative transform
/// stored in branch[0] node[1]
        const float * rt = (const float *)get_branch_node(rootBranch, 1);
        sgts[0] = rt[0];
        sgts[1] = rt[1];
        sgts[2] = rt[2];
        sgts[3] = rt[3];
        sgts[4] = rt[4];
        sgts[5] = rt[5];
/// get tight box
/// stored in branch[0] node[8]
        const float * tb = (const float *)get_branch_node(rootBranch, 8);
        sgtb[0] = tb[0];
        sgtb[1] = tb[1];
        sgtb[2] = tb[2];
        sgtb[3] = tb[3];
        sgtb[4] = tb[4];
        sgtb[5] = tb[5];
/// get grid box
/// stored in branch[0] node[4]
        const float * gb = (const float *)get_branch_node(rootBranch, 4);
        sgb[0] = gb[0];
        sgb[1] = gb[1];
        sgb[2] = gb[2];
        sgb[3] = gb[3];
        sgb[4] = gb[4];
        sgb[5] = gb[5];
        
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

/// transform into grid space
    v3_minusr<float4>(incident.o, sgts);
    v3_minusr<float4>(incident.d, sgts);
    v3_divider<float4>(incident.o, &sgts[3]);
    v3_divider<float4>(incident.d, &sgts[3]);
    
    v3_minus<float4, float4>(incident.d, incident.o);
    v3_normalize_inplace<float4>(incident.d);
    
    Aabb4 box;
/// test tight box
    aabb4_r(box, sgtb);
    
    incident.o.w = -1e28f;
    incident.d.w = 1e28f;
    
    float t0, t1;
    
    srayActive[tidx] = ray_box_slab1(t0, t1,
                            incident, box);
    __syncthreads();
    reduceSumInBlock<NumThreads, int>(tidx, srayActive);
	__syncthreads();
	
/// exit if no ray intersects tight box
    if(srayActive[0] < 1)
        return;
    
/// start with grid box
    aabb4_r(box, sgb);
	    
    float3 t0Normal, t1Normal;
    float3 shadingNormal = make_float3(0.f, 0.f, 0.f);
    
    for(;;) {
        srayActive[tidx] = ray_box_slab(t0, t1,
                            t0Normal, t1Normal, 
                            incident, box);
                            
        if(srayActive[tidx]) {
            shadingNormal = t0Normal;
            
            srayActive[tidx] = 0;
        }
        __syncthreads();
	
/// count active rays, result stored in srayActive[0]
	    reduceSumInBlock<NumThreads, int>(tidx, srayActive);
	
	    __syncthreads();
	    
/// exit if no active rays
	    if(srayActive[0] < 1)
	        break;
    }
    
    if(px < c_renderRect.x || px >= c_renderRect.z) return;
    if(py < c_renderRect.y || py >= c_renderRect.w) return;
    
	pix[getImagePixelIdx(px, py)] = encodeRGB(128 + 127 * shadingNormal.x,
                            128 + 127 * shadingNormal.y,
                            128 + 127 * shadingNormal.z);
	
}


