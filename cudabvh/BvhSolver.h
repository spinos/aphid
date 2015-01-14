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

class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;
class CudaLinearBvh;
class RayTest;

class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void setMesh(BvhTriangleMesh * mesh);
	void setRay(RayTest * ray);
	
	CudaLinearBvh * bvh();
	
	const bool isValid() const;

protected:
    virtual void stepPhysics(float dt);	

private:
	BvhTriangleMesh * m_mesh;
	CudaLinearBvh * m_bvh;
	RayTest * m_ray;
    
	bool m_isValid;
};
#endif        //  #ifndef BVHSOLVER_H
