/*
 *  SelectCondition.cpp
 *  aphid
 *
 *  Created by jian zhang on 11/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectCondition.h"

SelectCondition::SelectCondition() 
{
	m_byDistance = m_byRegion = m_byFacing = 1;
}

void SelectCondition::setDistanceFilter(char on)
{
	m_byDistance = on;
}

void SelectCondition::setRegionFilter(char on)
{
	m_byRegion = on;
}

void SelectCondition::setFacingFilter(char on)
{
	m_byFacing = on;
}
	
char SelectCondition::byDistance() const
{
	return m_byDistance;
}

char SelectCondition::byRegion() const
{
	return m_byRegion;
}

char SelectCondition::byFacing() const
{
	return m_byFacing;
}

char SelectCondition::filteredByFacing(const Vector3F & nor) const
{
	return nor.dot(normal) < 0.f;
}

char SelectCondition::filteredByDistance(const Vector3F & pos) const
{
	Vector3F d = pos - center;
	return d.length() > maxDistance;
}
