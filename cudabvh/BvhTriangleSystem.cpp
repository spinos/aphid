#include "BvhTriangleSystem.h"
#include <CUDABuffer.h>
#include "TriangleSystemInterface.h"
#include <CudaBase.h>
#include <masssystem_impl.h>

BvhTriangleSystem::BvhTriangleSystem(ATriangleMesh * md) : TriangleSystem(md)
{
}

BvhTriangleSystem::~BvhTriangleSystem() {}

void BvhTriangleSystem::initOnDevice() 
{
    std::cout<<"\n triangle system init on device";
	setNumPrimitives(numTriangles());
    CudaMassSystem::initOnDevice();
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
	void * vsrc = deviceVa();
    void * idx = deviceTretradhedronIndices();
    void * dst = leafAabbs();
    trianglesys::formTetrahedronAabbs((Aabb *)dst, (float3 *)cvs, (float3 *)vsrc, 1.f/60.f, (uint4 *)idx, numTriangles());
    CudaBase::CheckCudaError("triangle system form aabb");
}

void BvhTriangleSystem::integrate(float dt)
{
    masssystem::integrateAllAnchored((float3 *)deviceX(), 
                           (float3 *)deviceV(), 
                           (float3 *)deviceVa(), 
                           dt, 
                           numPoints());
    CudaBase::CheckCudaError("triangle system integrate");
}
//:~