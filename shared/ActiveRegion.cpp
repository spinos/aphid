/*
 *  ActiveRegion.cpp
 *  aphid
 *
 *  Created by jian zhang on 11/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "ActiveRegion.h"
namespace aphid {

ActiveRegion::ActiveRegion() {}
ActiveRegion::~ActiveRegion() 
{
	clearActiveRegion();
}

unsigned ActiveRegion::numActiveRegionFaces() const
{
	return m_regionFaces.size();
}

unsigned ActiveRegion::activeRegionFace(unsigned idx) const
{
	return m_regionFaces[idx];
}

char ActiveRegion::hasActiveRegion() const
{
    return m_regionFaces.size() > 0;
}

void ActiveRegion::clearActiveRegion()
{
    m_regionFaces.clear();
}

void ActiveRegion::addActiveRegionFace(unsigned idx)
{
	m_regionFaces.push_back(idx);
}

void ActiveRegion::resetActiveRegion() {}

}
