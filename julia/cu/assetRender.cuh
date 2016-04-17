#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include "cu/NTreeTraverse.cuh"
#include <cu/VoxelPrim.cuh>
#include "cuSMem.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

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
/// smem layout in floats
/// 0  -> 7     grid translation and scaling
/// 8  -> 15    grid tight box
/// 16 -> 23    grid box

    const int tidx = threadIdx.x + blockDim.x * threadIdx.y;
    if(tidx < 1) {
        const NTreeBranch4 & rootBranch = branches[0];
/// get relative transform
/// stored in branch[0] node[1]
        const float * rt = (const float *)get_branch_node(rootBranch, 1);
        sdata[0] = rt[0];
        sdata[1] = rt[1];
        sdata[2] = rt[2];
        sdata[3] = rt[3];
        sdata[4] = rt[4];
        sdata[5] = rt[5];
/// get tight box
/// stored in branch[0] node[8]
        const float * tb = (const float *)get_branch_node(rootBranch, 8);
        sdata[8] = tb[0];
        sdata[9] = tb[1];
        sdata[10] = tb[2];
        sdata[11] = tb[3];
        sdata[12] = tb[4];
        sdata[13] = tb[5];
/// get grid box
/// stored in branch[0] node[4]
        const float * gb = (const float *)get_branch_node(rootBranch, 4);
        sdata[16] = gb[0];
        sdata[17] = gb[1];
        sdata[18] = gb[2];
        sdata[19] = gb[3];
        sdata[20] = gb[4];
        sdata[21] = gb[5];
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
    v3_minusr<float4>(incident.o, &sdata[0]);
    v3_minusr<float4>(incident.d, &sdata[0]);
    v3_divider<float4>(incident.o, &sdata[3]);
    v3_divider<float4>(incident.d, &sdata[3]);
    
    v3_minus<float4, float4>(incident.d, incident.o);
    v3_normalize_inplace<float4>(incident.d);
    
    Aabb4 box;
/// test tight box
    aabb4_r(box, &sdata[8]);
    
    incident.o.w = -1e28f;
    incident.d.w = 1e28f;
    
    float t0, t1;
    float3 t0Normal, t1Normal;
    float3 shadingNormal = make_float3(0.f, 0.f, 0.f);
    if(ray_box_slab(t0, t1,
                    t0Normal, t1Normal, 
                    incident, box) ) {
        
            shadingNormal = t0Normal;

	}
	
    if(px < c_renderRect.x || px >= c_renderRect.z) return;
    if(py < c_renderRect.y || py >= c_renderRect.w) return;
    
	pix[getImagePixelIdx(px, py)] = encodeRGB(128 + 127 * shadingNormal.x,
                            128 + 127 * shadingNormal.y,
                            128 + 127 * shadingNormal.z);
	
}


