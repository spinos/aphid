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
	return nor.dot(m_normal) < 0.f;
}

char SelectCondition::filteredByDistance(const Vector3F & pos) const
{
	Vector3F d = pos - m_center;
	return d.length() > m_maxDistance;
}

char SelectCondition::filteredByProbability() const
{
	if(m_probability == 1.f) return 0;
	float r = ((float)(rand() % 199))/199.f;
	return r > m_probability;
}

void SelectCondition::setProbability(float p)
{
	m_probability = p;
}

void SelectCondition::setCenter(Vector3F c)
{
	m_center = c;
}

void SelectCondition::setNormal(Vector3F n)
{
	m_normal = n;
}
	
void SelectCondition::setMaxDistance(float d)
{
	m_maxDistance = d;
}

Vector3F SelectCondition::center() const
{
	return m_center;
}

Vector3F SelectCondition::normal() const
{
	return m_normal;
}

float SelectCondition::maxDistance() const
{
	return m_maxDistance;
}
