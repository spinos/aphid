#include "cu/ImageBase.cuh"
#include "cu/VectorMath.cuh"
#include "cu/RayIntersection.cuh"
#include "cu/NTreeTraverse.cuh"

__constant__ float3 c_frustumVec[6];
__constant__ int4 c_renderRect;

__global__ void twoCube_kernel(uint * pix, 
                                float * nearDepth,
                                float * farDepth,
                                NTreeBranch4 * branches,
                                NTreeLeaf * leaves,
                                Rope * ropes,
                                int * indirections,
                                Cube * primitives)
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
    aabb4_convert<Rope>(box, ropes[0]);
    
    uint ind = getTiledPixelIdx();
    
    incident.o.w = nearDepth[ind];
    incident.d.w = farDepth[ind];
    
    float tmin, tmax;
    if(!ray_box(incident, box, tmin, tmax) ) 
        return;
    
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

