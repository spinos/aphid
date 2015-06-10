#ifndef BVHTETRAHEDRONSYSTEM_H
#define BVHTETRAHEDRONSYSTEM_H

/*
 *  BvhTetrahedronSystem.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "CudaLinearBvh.h"
#include <TetrahedronSystem.h>
class CUDABuffer; 
class BvhTetrahedronSystem : public TetrahedronSystem, public CudaLinearBvh {
public:
	BvhTetrahedronSystem();
	BvhTetrahedronSystem(ATetrahedronMesh * md);
	virtual ~BvhTetrahedronSystem();
	virtual void initOnDevice();
	virtual void update();
	
	void * vicinity();
	
	virtual void integrate(float dt);
	void sendXToHost();
	void sendVToHost();
protected:
    
private:
	void formTetrahedronAabbs();
private:
	CUDABuffer * m_deviceTetrahedronVicinityInd;
	CUDABuffer * m_deviceTetrahedronVicinityStart;
	CUDABuffer * m_vicinity;
};
#endif        //  #ifndef CUDATETRAHEDRONSYSTEM_H
