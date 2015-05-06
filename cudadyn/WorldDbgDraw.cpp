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
