#include "WorldDbgDraw.h"
#include <GeoDrawer.h>
#include <CudaLinearBvh.h>
#include <radixsort_implement.h>

int WorldDbgDraw::MaxDrawBvhHierarchyLevel = 2;

WorldDbgDraw::WorldDbgDraw(GeoDrawer * drawer) 
{ m_drawer = drawer; }

WorldDbgDraw::~WorldDbgDraw() {}

#if DRAW_BVH_HASH

void WorldDbgDraw::showBvhHash(CudaLinearBvh * bvh)
{
    const unsigned n = bvh->numLeafNodes();
	Aabb * boxes = (Aabb *)bvh->hostLeafBox();
	BoundingBox bb;
	KeyValuePair * bvhHash = (KeyValuePair *)bvh->hostLeafHash();
	
	float red;
	Vector3F p, q;
	for(unsigned i=1; i < n; i++) {
		red = (float)i/(float)n;
		
		glColor3f(red, 1.f - red, 0.f);
		Aabb a0 = boxes[bvhHash[i-1].value];
		p.set(a0.low.x * 0.5f + a0.high.x * 0.5f, a0.low.y * 0.5f + a0.high.y * 0.5f + 0.2f, a0.low.z * 0.5f + a0.high.z * 0.5f);
		Aabb a1 = boxes[bvhHash[i].value];
		q.set(a1.low.x * 0.5f + a1.high.x * 0.5f, a1.low.y * 0.5f + a1.high.y * 0.5f + 0.2f, a1.low.z * 0.5f + a1.high.z * 0.5f);
        
#if DRAW_BVH_HASH_SFC       
		m_drawer->arrow(p, q);
#else
        bb.setMin(a0.low.x, a0.low.y, a0.low.z);
        bb.setMax(a0.high.x, a0.high.y, a0.high.z);
        m_drawer->boundingBox(bb);
#endif
	}
}
#endif

#if DRAW_BVH_HIERARCHY

inline int isLeafNode(int index) 
{ return (index >> 31 == 0); }

inline int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }

void WorldDbgDraw::showBvhHierarchy(CudaLinearBvh * bvh)
{
    const int rootNodeInd = 0x80000000;
	
	Aabb * internalBoxes = (Aabb *)bvh->hostInternalAabb();
	int2 * internalNodeChildIndices = (int2 *)bvh->hostInternalChildIndices();
	int * distanceFromRoot = (int *)bvh->hostInternalDistanceFromRoot();
	Aabb * primitiveBoxes = (Aabb *)bvh->hostPrimitiveAabb();
	KeyValuePair * primitiveHash = (KeyValuePair *)bvh->hostPrimitiveHash();
	
	BoundingBox bb;
	Aabb bvhNodeAabb;
	int stack[128];
	stack[0] = rootNodeInd;
	int stackSize = 1;
	int touchedInternal = 0;
	int level;
	int2 child;
	while(stackSize > 0) {
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		level = distanceFromRoot[bvhNodeIndex];
		if(level > MaxDrawBvhHierarchyLevel) continue;
		
		child = internalNodeChildIndices[bvhNodeIndex];
		bvhNodeAabb = internalBoxes[bvhNodeIndex];

		if(isLeafNode(child.x) || level >= MaxDrawBvhHierarchyLevel-1)
		 {
			bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
			bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
		
			m_drawer->setGroupColorLight(touchedInternal);
			m_drawer->boundingBox(bb);
		}

		touchedInternal++;
		
		if(isLeafNode(child.x)) {
		    drawPrimitiveBoxes(primitiveBoxes, primitiveHash, 
		        child.x,
		        child.y);
		    continue;
		}
		
		if(stackSize + 2 > 128)
		{
			//Error
		}
		else
		{
			stack[ stackSize ] = child.x;
			stackSize++;
			stack[ stackSize ] = child.y;
			stackSize++;
		}
			
	}
}
#endif

void WorldDbgDraw::drawPrimitiveBoxes(void * boxes, void * indirections, 
        int begin, int end)
{
    Aabb * primitiveBox = (Aabb *)boxes; 
    KeyValuePair * primitiveHash = (KeyValuePair *)indirections;
    Aabb ab;
    BoundingBox bb;
	int i = begin;
    for(;i<=end;i++) {
        ab = primitiveBox[primitiveHash[i].value];
        bb.setMin(ab.low.x- 1e-3, ab.low.y- 1e-3, ab.low.z- 1e-3);
		bb.setMax(ab.high.x+ 1e-3, ab.high.y+ 1e-3, ab.high.z+ 1e-3);
		
		m_drawer->boundingBox(bb);
    }
}

