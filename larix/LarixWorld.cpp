/*
 *  LarixWorld.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LarixWorld.h"
#include <ATetrahedronMesh.h>
#include <APointCloud.h>
#include <AdaptiveGrid.h>

LarixWorld::LarixWorld() 
{ 
    m_cloud = 0;
    m_mesh = 0;
	m_grid = 0;
}

LarixWorld::~LarixWorld() 
{ 
    if(m_cloud) delete m_cloud; 
    if(m_mesh) delete m_mesh;
	if(m_grid) delete m_grid;
}

void LarixWorld::setTetrahedronMesh(ATetrahedronMesh * m)
{ m_mesh = m; }

ATetrahedronMesh * LarixWorld::tetrahedronMesh() const
{ return m_mesh; }

void LarixWorld::setPointCloud(APointCloud * pc)
{ m_cloud = pc; }

APointCloud * LarixWorld::pointCloud() const
{ return m_cloud; }

void LarixWorld::setGrid(AdaptiveGrid * g)
{ m_grid = g; }

AdaptiveGrid * LarixWorld::grid() const
{ return m_grid; }
//:~