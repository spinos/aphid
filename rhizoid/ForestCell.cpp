/*
 *  ForestCell.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ForestCell.h"
#include <PlantCommon.h>
#include <sdb/LodSampleCache.h>

namespace aphid {

ForestCell::ForestCell(Entity * parent) : sdb::Array<sdb::Coord2, Plant>(parent)
{
	m_lodsamp = new sdb::LodSampleCache;
}

ForestCell::~ForestCell()
{
	delete m_lodsamp;
}

}
