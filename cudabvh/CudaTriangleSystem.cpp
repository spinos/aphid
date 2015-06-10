#include "CudaTriangleSystem.h"
#include <CUDABuffer.h>
#include "TriangleSystemInterface.h"
CudaTriangleSystem::CudaTriangleSystem(ATriangleMesh * md) : TriangleSystem(md)
{
}

CudaTriangleSystem::~CudaTriangleSystem() {}

void CudaTriangleSystem::initOnDevice() 
{
	setNumPrimitives(numTriangles());
	CudaLinearBvh::initOnDevice();
}

void CudaTriangleSystem::update()
{
	formTetrahedronAabbs();
    CudaLinearBvh::update();
}

void CudaTriangleSystem::formTetrahedronAabbs()
{
    void * cvs = deviceX();
	void * vsrc = deviceV();
    void * idx = deviceTretradhedronIndices();
    void * dst = leafAabbs();
    trianglesys::formTetrahedronAabbs((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTriangles());
}
//:~