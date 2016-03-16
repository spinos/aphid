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

inline __device__ int is_leaf(const KdNode & node)
{
    return ( node.leaf.combined & 0x4 ) > 0;
}

inline __device__ int get_prim_offset(const KdNode & node)
{
    return ( node.leaf.combined & 0x4 ) >> 3;
}

inline __device__ int get_prim_length(const KdNode & node)
{
    return node.leaf.end;
}

inline __device__ int get_split_axis(const KdNode & node)
{ 
	return node.inner.combined & 0x3; 
}

inline __device__ float get_split_pos(const KdNode & node)
{ 
	return node.inner.split; 
}

inline __device__ void decode_rope(const int & src, int & itreelet, int & inode)
{
	itreelet = src >> 5;
	inode = src & ((1<<5)-1 );
}


#endif        //  #ifndef NTREETRAVERSE_CUH

