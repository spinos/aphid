#include "CudaTriangleSystem.h"
#include <CUDABuffer.h>
#include "TriangleSystemInterface.h"
CudaTriangleSystem::CudaTriangleSystem(ATriangleMesh * md) : TriangleSystem(md)  {}
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

void CudaTriangleSystem::setDeviceXPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceX = ptr; m_xLoc = loc; }

void CudaTriangleSystem::setDeviceXiPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceXi = ptr; m_xiLoc = loc; }

void CudaTriangleSystem::setDeviceVPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceV = ptr; m_vLoc = loc; }

void CudaTriangleSystem::setDeviceMassPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceMass = ptr; m_massLoc = loc; }

void CudaTriangleSystem::setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc)
{ m_deviceTetrahedronIndices = ptr; m_iLoc = loc; }

void * CudaTriangleSystem::deviceX()
{ return m_deviceX->bufferOnDeviceAt(m_xLoc); }

void * CudaTriangleSystem::deviceXi()
{ return m_deviceXi->bufferOnDeviceAt(m_xiLoc); }

void * CudaTriangleSystem::deviceV()
{  return m_deviceV->bufferOnDeviceAt(m_vLoc); }

void * CudaTriangleSystem::deviceMass()
{  return m_deviceMass->bufferOnDeviceAt(m_massLoc); }

void * CudaTriangleSystem::deviceTretradhedronIndices()
{ return m_deviceTetrahedronIndices->bufferOnDeviceAt(m_iLoc); }
