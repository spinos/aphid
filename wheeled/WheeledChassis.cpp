/*
 *  WheeledChassis.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "WheeledChassis.h"
namespace caterpillar {
WheeledChassis::WheeledChassis() 
{
	m_hullDim.set(1.f,1.f,1.f);
	m_numAxis = 0;
}

WheeledChassis::~WheeledChassis() {}

void WheeledChassis::setHullDim(const Vector3F & p) { m_hullDim = p; }
void WheeledChassis::setOrigin(const Vector3F & p) { m_origin = p; }
void WheeledChassis::setNumAxis(const int & x) 
{
	m_numAxis = x;
	if(m_numAxis < 2) m_numAxis = 2;
	if(m_numAxis > MAXNUMAXIS) m_numAxis = MAXNUMAXIS;
}

void WheeledChassis::setAxisCoord(const int & i, const float & span, const float & y, const float & z)
{
	m_axisCoord[i].set(span, y, z);
}

}