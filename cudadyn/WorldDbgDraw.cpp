#include "WorldDbgDraw.h"
#include <GeoDrawer.h>
#include <CudaLinearBvh.h>
#include <radixsort_implement.h>

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
    const int rootNodeInd = 0x80000000;//bvh->hostRootInd(); // std::cout<<" root "<< rootNodeInd;
	
    // const unsigned numInternal = bvh->numInternalNodes();
	
	Aabb * internalBoxes = (Aabb *)bvh->hostInternalAabb();
	int2 * internalNodeChildIndices = (int2 *)bvh->hostInternalChildIndices();
	// int * distanceFromRoot = (int *)bvh->hostInternalDistanceFromRoot();
	
	BoundingBox bb;
	Aabb bvhNodeAabb;
	int stack[128];
	stack[0] = rootNodeInd;
	int stackSize = 1;
	int maxStack = 1;
	int touchedInternal = 0;
	//int maxLevel = 0;
	//int level;
	while(stackSize > 0) {
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		//level = distanceFromRoot[bvhNodeIndex];
		//if(level > m_maxDisplayLevel) continue;
		//if(maxLevel < level)
			//maxLevel = level;
			
		bvhNodeAabb = internalBoxes[bvhNodeIndex];

		
		//if(m_maxDisplayLevel - level < 2) 
		if(isLeafNode(internalOrLeafNodeIndex)==0)
		 {
			bb.setMin(bvhNodeAabb.low.x, bvhNodeAabb.low.y, bvhNodeAabb.low.z);
			bb.setMax(bvhNodeAabb.high.x, bvhNodeAabb.high.y, bvhNodeAabb.high.z);
		
			m_drawer->setGroupColorLight(touchedInternal);
			m_drawer->boundingBox(bb);
		}

		touchedInternal++;
		
		if(isLeafNode(internalOrLeafNodeIndex)) continue;
		
		if(stackSize + 2 > 128)
		{
			//Error
		}
		else
		{
			stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
			stackSize++;
			stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
			stackSize++;
			
			if(stackSize > maxStack) maxStack = stackSize;
		}
			
	}
#if 0
	std::cout//<<" total n internal node "<<numInternal<<"\n"
		<<" n internal node reached "<<touchedInternal<<"\n"
		<<" max draw bvh hierarchy stack size "<<maxStack<<"\n"
		<<" max level reached "<<maxLevel<<"\n";
#endif
}
#endif
