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

#include "BvhSolver.h"
#include "createBvh_implement.h"
#include "traverseBvh_implement.h"
#include "reduceBox_implement.h"
#include "reduceRange_implement.h"

BvhSolver::BvhSolver(QObject *parent) : BaseSolverThread(parent) 
{
	m_alpha = 0;
	m_isValid = 0;
	m_bvh = new CudaLinearBvh;
}

BvhSolver::~BvhSolver() {}

void BvhSolver::setMesh(BvhTriangleMesh * mesh)
{ 
	m_mesh = mesh;
	m_bvh->setMesh(mesh); 
}

void BvhSolver::createRays(uint m, uint n)
{
	m_numRays = m * n;
	m_rayDim = m;
	m_rays = new CUDABuffer;
	m_rays->create(m_numRays * sizeof(RayInfo));
	
	m_ntests = new CUDABuffer;
	m_ntests->create(m_numRays * sizeof(float));
}

void BvhSolver::stepPhysics(float dt)
{
	m_mesh->update();
	m_bvh->update();
	formRays();
	rayTraverse();
	m_isValid = 1;
}

void BvhSolver::formRays()
{
	void * rays = m_rays->bufferOnDevice();
	
	float3 ori; 
	ori.x = sin(m_alpha * 0.2f) * 60.f;
	ori.y = 60.f + sin(m_alpha * 0.1f) * 20.f;
	ori.z = cos(m_alpha * 0.2f) * 30.f;
	bvhTestRay((RayInfo *)rays, ori, 10.f, m_rayDim, m_numRays);
}

void BvhSolver::rayTraverse()
{
	void * rays = m_rays->bufferOnDevice();
	void * rootNodeIndex = m_bvh->rootNodeIndex();
	void * internalNodeChildIndex = m_bvh->internalNodeChildIndices();
	void * internalNodeAabbs = m_bvh->internalNodeAabbs();
	void * leafNodeAabbs = m_bvh->leafAabbs();
	void * mortonCodesAndAabbIndices = m_bvh->leafHash();
	void * o_nts = m_ntests->bufferOnDevice();
	bvhRayTraverseIterative((RayInfo *)rays,
								(int *)rootNodeIndex, 
								(int2 *)internalNodeChildIndex, 
								(Aabb *)internalNodeAabbs, 
								(Aabb *)leafNodeAabbs,
								(KeyValuePair *)mortonCodesAndAabbIndices,								
								(float *) o_nts,
								numRays());
}

void BvhSolver::setAlpha(float x) 
{ m_alpha = x; }

const unsigned BvhSolver::numRays() const 
{ return m_numRays; }

void BvhSolver::getRays(BaseBuffer * dst) 
{ m_rays->deviceToHost(dst->data(), m_rays->bufferSize()); }

CudaLinearBvh * BvhSolver::bvh()
{ return m_bvh; }

const bool BvhSolver::isValid() const
{ return m_isValid; }
