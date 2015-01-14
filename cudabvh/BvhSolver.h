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
#include <bvh_common.h>
#include <radixsort_implement.h>

class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;
class CudaLinearBvh;

class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void setMesh(BvhTriangleMesh * mesh);
	void createRays(uint m, uint n);
	
	const unsigned numRays() const;
	
	void setAlpha(float x);
	void getRays(BaseBuffer * dst);
	CudaLinearBvh * bvh();
	
	const bool isValid() const;

protected:
    virtual void stepPhysics(float dt);	
private:
	void formRays();
	void rayTraverse();
	
private:
	CUDABuffer * m_rays;
	CUDABuffer * m_ntests;
	BvhTriangleMesh * m_mesh;
	CudaLinearBvh * m_bvh;
    
	unsigned m_numRays, m_rayDim;
	float m_alpha;
	bool m_isValid;
};
#endif        //  #ifndef BVHSOLVER_H
