#ifndef BVHSOLVER_H
#define BVHSOLVER_H

/*
 *  BvhSolver.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <BaseSolverThread.h>
#include <app_define.h>

class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;
class CudaLinearBvh;
class RayTest;
class CudaParticleSystem;

class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void setMesh(BvhTriangleMesh * mesh);
	void setRay(RayTest * ray);
	void setParticleSystem(CudaParticleSystem * particles);
	
	const bool isValid() const;
	
#ifdef BVHSOLVER_DBG_DRAW
    void setHostPtrs(BaseBuffer * leafAabbs,
                    BaseBuffer * internalAabbs,
                    BaseBuffer * internalDistance,
                    BaseBuffer * leafHash,
                    BaseBuffer * internalChildIndices,
                    int * rootNodeInd);
#endif

protected:
    virtual void stepPhysics(float dt);	

private:
#ifdef BVHSOLVER_DBG_DRAW
    void sendDataToHost();
#endif
private:
	BvhTriangleMesh * m_mesh;
	RayTest * m_ray;
	CudaParticleSystem * m_particles;
	
#ifdef BVHSOLVER_DBG_DRAW
    BaseBuffer * m_hostLeafAabbs;
	BaseBuffer * m_hostInternalAabbs;
	BaseBuffer * m_hostInternalDistance;
	BaseBuffer * m_hostLeafHash;
	BaseBuffer * m_hostInternalChildIndices;
	int * m_hostRootNodeInd;
#endif
    
	bool m_isValid;
};
#endif        //  #ifndef BVHSOLVER_H
