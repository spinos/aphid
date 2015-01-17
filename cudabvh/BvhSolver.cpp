/*
 *  BvhSolver.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtCore>
#include <CudaBase.h>
#include <CUDABuffer.h>
#include "BvhTriangleMesh.h"
#include "CudaLinearBvh.h"
#include "rayTest.h"
#include "CudaParticleSystem.h"

#include "BvhSolver.h"
#include "createBvh_implement.h"
#include "traverseBvh_implement.h"
#include "reduceBox_implement.h"
#include "reduceRange_implement.h"

BvhSolver::BvhSolver(QObject *parent) : BaseSolverThread(parent) 
{
	m_isValid = 0;
	// m_bvh = new CudaLinearBvh;
}

BvhSolver::~BvhSolver() {}

void BvhSolver::setMesh(BvhTriangleMesh * mesh)
{ 
	m_mesh = mesh;
	// m_bvh->setNumLeafNodes(mesh->numTriangles());
	// m_bvh->create(); 
}

void BvhSolver::setRay(RayTest * ray)
{ 
    m_ray = ray;
    m_ray->setBvh(m_mesh->bvh());
}

void BvhSolver::setParticleSystem(CudaParticleSystem * particles)
{
    m_particles = particles;
}

void BvhSolver::stepPhysics(float dt)
{
	m_mesh->update();
	// m_bvh->update();
	m_ray->update();
	m_particles->update(dt);
#ifdef BVHSOLVER_DBG_DRAW
    cudaDeviceSynchronize();
    sendDataToHost();
#endif
	m_isValid = 1;
}

const bool BvhSolver::isValid() const
{ return m_isValid; }

#ifdef BVHSOLVER_DBG_DRAW
void BvhSolver::setHostPtrs(BaseBuffer * leafAabbs,
                BaseBuffer * internalAabbs,
                BaseBuffer * internalDistance,
                BaseBuffer * leafHash,
                BaseBuffer * internalChildIndices,
                int * rootNodeInd)
{
    m_hostLeafAabbs = leafAabbs;
    m_hostInternalAabbs = internalAabbs;
    m_hostInternalDistance = internalDistance;
    m_hostLeafHash = leafHash;
    m_hostInternalChildIndices = internalChildIndices;
    m_hostRootNodeInd = rootNodeInd;
}

void BvhSolver::sendDataToHost()
{
    m_mesh->deviceToHost();
    m_mesh->bvh()->getLeafAabbs(m_hostLeafAabbs);
	m_mesh->bvh()->getInternalAabbs(m_hostInternalAabbs);
    m_mesh->bvh()->getInternalDistances(m_hostInternalDistance);
	m_mesh->bvh()->getLeafHash(m_hostLeafHash);
	m_mesh->bvh()->getInternalChildIndex(m_hostInternalChildIndices);
	m_mesh->bvh()->getRootNodeIndex(m_hostRootNodeInd);
	m_particles->deviceToHost();
}

#endif

