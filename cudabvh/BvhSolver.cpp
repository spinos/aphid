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

#include "BvhSolver.h"
#include "createBvh_implement.h"
#include "traverseBvh_implement.h"
#include "reduceBox_implement.h"
#include "reduceRange_implement.h"

BvhSolver::BvhSolver(QObject *parent) : BaseSolverThread(parent) 
{
	m_isValid = 0;
	m_bvh = new CudaLinearBvh;
}

BvhSolver::~BvhSolver() {}

void BvhSolver::setMesh(BvhTriangleMesh * mesh)
{ 
	m_mesh = mesh;
	m_bvh->setMesh(mesh); 
}

void BvhSolver::setRay(RayTest * ray)
{ 
    m_ray = ray;
    m_ray->setBvh(m_bvh);
}

void BvhSolver::stepPhysics(float dt)
{
	m_mesh->update();
	m_bvh->update();
	m_ray->update();
	m_isValid = 1;
}

CudaLinearBvh * BvhSolver::bvh()
{ return m_bvh; }

const bool BvhSolver::isValid() const
{ return m_isValid; }

