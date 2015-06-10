#include "BvhTriangleSystem.h"
#include <CUDABuffer.h>
#include "TriangleSystemInterface.h"
BvhTriangleSystem::BvhTriangleSystem(ATriangleMesh * md) : TriangleSystem(md)
{
}

BvhTriangleSystem::~BvhTriangleSystem() {}

void BvhTriangleSystem::initOnDevice() 
{
	setNumPrimitives(numTriangles());
	CudaLinearBvh::initOnDevice();
}

void BvhTriangleSystem::update()
{
	formTetrahedronAabbs();
    CudaLinearBvh::update();
}

void BvhTriangleSystem::formTetrahedronAabbs()
{
    void * cvs = deviceX();
	void * vsrc = deviceV();
    void * idx = deviceTretradhedronIndices();
    void * dst = leafAabbs();
    trianglesys::formTetrahedronAabbs((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTriangles());
}
//:~