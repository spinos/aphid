#ifndef CUDATRIANGLESYSTEM_H
#define CUDATRIANGLESYSTEM_H
#include "ATriangleMesh.h"
#include "TriangleSystem.h"
#include "CudaLinearBvh.h"
class CUDABuffer;

class CudaTriangleSystem : public TriangleSystem, public CudaLinearBvh {
public:
    CudaTriangleSystem(ATriangleMesh * md);
    virtual ~CudaTriangleSystem();

    virtual void initOnDevice();
	virtual void update();
    
    void setDeviceXPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceXiPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceVPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceMassPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc);

    void * deviceX();
	void * deviceXi();
	void * deviceV();
	void * deviceMass();
	void * deviceTretradhedronIndices();
protected:
private:
	void formTetrahedronAabbs();
private:
    CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceXi;
	CUDABuffer * m_deviceV;
	CUDABuffer * m_deviceMass;
	CUDABuffer * m_deviceTetrahedronIndices;
	unsigned m_xLoc, m_xiLoc, m_vLoc, m_massLoc, m_iLoc;
};

#endif        //  #ifndef CUDATRIANGLESYSTEM_H

