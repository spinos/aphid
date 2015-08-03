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
#include <AdaptiveField.h>

LarixWorld::LarixWorld() 
{ 
    m_cloud = 0;
    m_mesh = 0;
	m_field = 0;
}

LarixWorld::~LarixWorld() 
{ 
    if(m_cloud) delete m_cloud; 
    if(m_mesh) delete m_mesh;
	if(m_field) delete m_field;
}

void LarixWorld::setTetrahedronMesh(ATetrahedronMesh * m)
{ m_mesh = m; }

ATetrahedronMesh * LarixWorld::tetrahedronMesh() const
{ return m_mesh; }

void LarixWorld::setPointCloud(APointCloud * pc)
{ m_cloud = pc; }

APointCloud * LarixWorld::pointCloud() const
{ return m_cloud; }

void LarixWorld::setField(AdaptiveField * g)
{ m_field = g; }

AdaptiveField * LarixWorld::field() const
{ return m_field; }
//:~