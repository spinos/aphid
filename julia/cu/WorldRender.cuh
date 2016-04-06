#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include "cu/NTreeTraverse.cuh"
#include "cuSMem.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

__global__ void worldBox_kernel(uint * pix, 
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

    if(threadIdx.x ==0 && threadIdx.y ==0) {
        const NTreeBranch4 & rootBranch = branches[0];
/// get relative transform
/// stored in branch[0] node[1]
///        const float * rt = (const float *)get_branch_node(rootBranch, 1);
///        sdata[0] = rt[0];
///        sdata[1] = rt[1];
///        sdata[2] = rt[2];
///        sdata[3] = rt[3];
///        sdata[4] = rt[4];
///        sdata[5] = rt[5];
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
    
    if(px < c_renderRect.x || px >= c_renderRect.z) return;
    if(py < c_renderRect.y || py >= c_renderRect.w) return;
    
    Ray4 incident;
    
    v3_convert<float4, float3>(incident.o, c_frustumVec[0]);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[1], px);
    v3_add_mult<float4, float3, uint>(incident.o, c_frustumVec[2], py);
    
    v3_convert<float4, float3>(incident.d, c_frustumVec[3]);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[4], px);
    v3_add_mult<float4, float3, uint>(incident.d, c_frustumVec[5], py);

/// transform into grid space
/// not necessary for world
///    v3_minusr<float4>(incident.o, &sdata[0]);
///    v3_minusr<float4>(incident.d, &sdata[0]);
///    v3_divider<float4>(incident.o, &sdata[3]);
///    v3_divider<float4>(incident.d, &sdata[3]);
    
    v3_minus<float4, float4>(incident.d, incident.o);
    v3_normalize_inplace<float4>(incident.d);
    
    Aabb4 box;
/// test tight box
    aabb4_r(box, &sdata[8]);
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = nearDepth[ind];
    incident.d.w = farDepth[ind];
    
    float tmin, tmax;
    if(!ray_box(incident, box, tmin, tmax) ) 
        return;
    
/// start with grid box
    aabb4_r(box, &sdata[16]);

    const KdNode * kn = get_branch_node(branches[0], 0);
    
    if(is_leaf(kn) ) return;
    
    int branchIdx = get_inner_offset(kn) & ~(1<<20);
	int nodeIdx = first_visit(kn, incident, box);
    
    int hasNext = 1;
    int stat;
    while (hasNext) {
		stat = visit_leaf(box, incident, get_branch_node(branches[branchIdx], nodeIdx), 
		                    branchIdx, nodeIdx);
		if(stat > 0 ) {
            if(hit_primitive(box, 
                                incident, 
                                get_branch_node(branches[branchIdx], nodeIdx),
                                leaves,
                                indirections,
                                primitives) )
			    hasNext = 0;
            else
                stat = 0;
		}
        if(stat == 0) {
		    hasNext = climb_rope(box, incident, 
		                    leaves,
		                    ropes, 
		                    get_branch_node(branches[branchIdx], nodeIdx),
		                    branchIdx, nodeIdx);
		}
	}

/// empty
	if(stat < 1) 
	    return;

	ray_box(incident, box, tmin, tmax);
	float3 hitP;
    ray_progress(hitP, incident, tmin - 1e-5f);
    float3 hitN = c_ray_box_face[side1_on_aabb4<float4>(box, hitP)];
	
    int r = 128 + 127 * hitN.x;
    int g = 128 + 127 * hitN.y;
    int b = 128 + 127 * hitN.z;
    
	pix[ind] = encodeRGB(r, g, b);
	
}

