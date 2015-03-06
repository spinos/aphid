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
	
	void setDeviceXPtr(void * ptr);
	void setDeviceVPtr(void * ptr);
	void setDeviceTretradhedronIndicesPtr(void * ptr);

	void * deviceX();
	void * deviceV();
	void * deviceTretradhedronIndices();
	
protected:
    
private:
	void formTetrahedronAabbs();
private:
	void * m_deviceX;
	void * m_deviceV;
	void * m_deviceTretradhedronIndices;
};
#endif        //  #ifndef CUDATETRAHEDRONSYSTEM_H
