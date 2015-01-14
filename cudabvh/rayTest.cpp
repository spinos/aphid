#include <CudaLinearBvh.h>
#include <CUDABuffer.h>
#include "rayTest.h"
#include "traverseBvh_implement.h"
#include <math.h>

RayTest::RayTest() {}
RayTest::~RayTest() {}
	
void RayTest::setBvh(CudaLinearBvh * bvh)
{ m_bvh = bvh; }

void RayTest::createRays(uint m, uint n)
{
    m_numRays = m * n;
	m_rayDim = m;
	m_rays = new CUDABuffer;
	m_rays->create(m_numRays * sizeof(RayInfo));
}

const unsigned RayTest::numRays() const
{ return m_numRays; }

void RayTest::setAlpha(float x)
{ m_alpha = x; }

void RayTest::getRays(BaseBuffer * dst)
{ m_rays->deviceToHost(dst->data(), m_rays->bufferSize()); }


void RayTest::update()
{
    formRays();
    rayTraverse();
}

void RayTest::formRays()
{
    void * rays = m_rays->bufferOnDevice();
	
	float3 ori; 
	ori.x = sin(m_alpha * 0.2f) * 60.f;
	ori.y = 60.f + sin(m_alpha * 0.1f) * 20.f;
	ori.z = cos(m_alpha * 0.2f) * 30.f;
	bvhTestRay((RayInfo *)rays, ori, 10.f, m_rayDim, m_numRays);
}

void RayTest::rayTraverse()
{
    void * rays = m_rays->bufferOnDevice();
	void * rootNodeIndex = m_bvh->rootNodeIndex();
	void * internalNodeChildIndex = m_bvh->internalNodeChildIndices();
	void * internalNodeAabbs = m_bvh->internalNodeAabbs();
	void * leafNodeAabbs = m_bvh->leafAabbs();
	void * mortonCodesAndAabbIndices = m_bvh->leafHash();
	bvhRayTraverseIterative((RayInfo *)rays,
								(int *)rootNodeIndex, 
								(int2 *)internalNodeChildIndex, 
								(Aabb *)internalNodeAabbs, 
								(Aabb *)leafNodeAabbs,
								(KeyValuePair *)mortonCodesAndAabbIndices,								
								numRays());
}

