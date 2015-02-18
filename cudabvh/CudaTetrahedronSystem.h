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
protected:
	void * deviceX();
	void * deviceV();
	void * deviceTretradhedronIndices();
private:
	void formTetrahedronAabbs();
private:
	CUDABuffer * m_deviceX;
	CUDABuffer * m_deviceV;
	CUDABuffer * m_deviceTretradhedronIndices;
};
#endif        //  #ifndef CUDATETRAHEDRONSYSTEM_H
