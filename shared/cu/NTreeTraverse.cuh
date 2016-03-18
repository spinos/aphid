#ifndef NTREETRAVERSE_CUH
#define NTREETRAVERSE_CUH

#include "RayIntersection.cuh"

struct KdNode {
    union {
			struct {
				int combined;
				float split;
			} inner;

			struct {
				int combined;
				int end;
			} leaf;
    };
};

struct Rope {
    float3 low;
    float3 high;
    int pad0;
    int treeletNode;
};

struct NTreeBranch4 {
    KdNode _node[30];
    float4 nouse;
};

struct NTreeLeaf {
    int _ropeInd[6];
	int _primStart;
	int _primLength;
};

inline __device__ int is_leaf(const KdNode * node)
{
    return ( node->leaf.combined & 0x4 ) > 0;
}

inline __device__ int get_prim_offset(const KdNode * node)
{
    return node->leaf.combined >> 3;
}

inline __device__ int get_prim_length(const KdNode * node)
{
    return node->leaf.end;
}

inline __device__ int get_split_axis(const KdNode * node)
{ 
	return node->inner.combined & 0x3; 
}

inline __device__ int get_inner_offset(const KdNode * node)
{
    return node->inner.combined >> 3;
}

inline __device__ float get_split_pos(const KdNode * node)
{ 
	return node->inner.split; 
}

inline __device__ const KdNode * get_branch_node(const NTreeBranch4 & src, 
                                        const int & i)
{ return &src._node[i]; }

inline __device__ int get_branch_internal_offset(const NTreeBranch4 & src, 
                                        const int & i)
{ return get_inner_offset(&src._node[i]) & ~(1<<20); }

inline __device__ int first_visit(const KdNode * node, 
                                    const Ray4 & incident,
                                    Aabb4 & box)
{
    const int axis = get_split_axis(node);
	const float splitPos = get_split_pos(node);
	const float o = v3_component<float4>(incident.o, axis);
	const float d = v3_component<float4>(incident.d, axis);
	
	Aabb4 lftBox, rgtBox;
	aabb4_split(lftBox, rgtBox, box, axis, splitPos);
	
	int above = o >= splitPos;
	
	if(absoluteValueF(d) < 1e-5f) {
	    if(above) box = rgtBox;
		else box = lftBox;
		
		return above;
	}
	
	if(is_v3_inside<Aabb4, float4>(lftBox, incident.o) ) {
		box = lftBox;
		return 0;
	}
	
	if(is_v3_inside<Aabb4, float4>(rgtBox, incident.o) ) {
		box = rgtBox;
		return 1;
	}
	
	float t = (splitPos - o) / d;
	if(t < 0.f) {
		if(above) box = rgtBox;
		else box = lftBox;
		return above;
	}
	
	float3 hitP;
	ray_progress(hitP, incident, t);
	if(is_v3_inside<Aabb4, float3>(box, hitP) ) {
		if(above) box = rgtBox;
		else box = lftBox;
		return above;
	}
	
	Aabb4 p2h;
	aabb4_reset(p2h);
	aabb4_expand<float4>(p2h, incident.o);
	aabb4_expand<float3>(p2h, hitP);
	
	if( aabb4_touch(p2h, box) == 0) {
	    if(above) above = 0;
	    else above = 1;
	}
	
	if(above) box = rgtBox;
	else box = lftBox;
		
	return above;
}

inline __device__ int visit_leaf(Aabb4 & box, 
                                const Ray4 & incident, 
                                 const KdNode * r,
								 int & branchIdx, 
                                int & nodeIdx)
{
    if(is_leaf(r) ) {
        if(get_prim_length(r) < 1)
            return 0;
        
        return 1;
    }
    
    const int offset = get_inner_offset(r);
	if(offset < (1<<20) ) {
		nodeIdx += offset + first_visit(r, incident, box);
	}
	else {
		branchIdx += offset & (~(1<<20));
		nodeIdx = first_visit(r, incident, box);
	}
    return -1;
}

inline __device__ void decode_rope(const int & src, int & itreelet, int & inode)
{
	itreelet = src >> 5;
	inode = src & 31;
}

inline __device__ int climb_rope(Aabb4 & box, 
                            const Ray4 & incident, 
                            NTreeLeaf * leaves,
                            Rope * ropes, 
		                    const KdNode * r,
                            int & branchIdx, 
                            int & nodeIdx)
{
    float tmin, tmax;
    ray_box(incident, box, tmin, tmax);
    
    float3 hitP;
    ray_progress(hitP, incident, tmax + 1e-4f);
    int side = side1_on_aabb4<float4>(box, hitP);
    
    int iLeaf = get_prim_offset(r);
    int iRope = leaves[iLeaf]._ropeInd[side];
    if(iRope < 1) return 0;
    
    const Rope & rp = ropes[iRope];
    decode_rope(rp.treeletNode, branchIdx, nodeIdx);
    aabb4_convert<Rope>(box, rp); 
    return 1;   
}

inline __device__ int hit_primitive(Aabb4 & box,
                            Ray4 & incident, 
                            const KdNode * r,
                            NTreeLeaf * leaves,
                            int * indirections,
                            Cube * primitives)
{
    int iLeaf = get_prim_offset(r);
    const NTreeLeaf & lea = leaves[iLeaf];
    int i = lea._primStart;
    const int end = i + lea._primLength;
    float tmin, tmax;
    Aabb4 b;
    int nhit = 0;
    for(;i<end;++i) {
        aabb4_from_cube(b, primitives[indirections[i] ]);
        if(ray_box(incident, b, tmin, tmax) ) {
            box = b;
            incident.d.w = tmax;
            nhit++;
        }
    }
    return nhit;
}

#endif        //  #ifndef NTREETRAVERSE_CUH

