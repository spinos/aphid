#ifndef CUDATETRAHEDRONSYSTEM_H
#define CUDATETRAHEDRONSYSTEM_H

/*
 *  CudaTetrahedronSystem.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "TetrahedronSystem.h"
#include "CudaLinearBvh.h"
class CUDABuffer;
class CudaTetrahedronSystem : public TetrahedronSystem, public CudaLinearBvh {
public:
	CudaTetrahedronSystem();
	virtual ~CudaTetrahedronSystem();
	virtual void initOnDevice();
	virtual void update();
	
	void setDeviceXPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceVPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceMassPtr(CUDABuffer * ptr, unsigned loc);
	void setDeviceTretradhedronIndicesPtr(CUDABuffer * ptr, unsigned loc);

	void * deviceX();
	void * deviceV();
	void * deviceTretradhedronIndices();
	
	void integrate(float timeStep);
	void sendXToHost();
	void sendVToHost();
	
protected:
    
private:
	void formTetrahedronAabbs();
private:
	CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceV;
	CUDABuffer * m_deviceMass;
	CUDABuffer * m_deviceTretradhedronIndices;
	unsigned m_xLoc, m_vLoc, m_massLoc, m_iLoc;
};
#endif        //  #ifndef CUDATETRAHEDRONSYSTEM_H
