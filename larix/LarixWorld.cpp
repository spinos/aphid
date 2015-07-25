/*
 *  LarixWorld.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LarixWorld.h"
#include <APointCloud.h>

LarixWorld::LarixWorld() 
{ m_cloud = 0; }

LarixWorld::~LarixWorld() 
{ if(m_cloud) delete m_cloud; }

void LarixWorld::setPointCloud(APointCloud * pc)
{ m_cloud = pc; }

APointCloud * LarixWorld::pointCloud() const
{ return m_cloud; }