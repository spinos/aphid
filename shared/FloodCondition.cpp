/*
 *  FloodCondition.cpp
 *  aphid
 *
 *  Created by jian zhang on 11/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "FloodCondition.h"

FloodCondition::FloodCondition() 
{
	m_minDistance = 0.1f;
}

void FloodCondition::setMinDistance(float d) 
{
	m_minDistance = d;
}

float FloodCondition::minDistance() const
{
	return m_minDistance;
}