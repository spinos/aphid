/*
 *  SplitEvent.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "SplitEvent.h"

int SplitEvent::Dimension = 3;

SplitEvent::SplitEvent() 
{
}

SplitEvent::~SplitEvent() {}

void SplitEvent::clear()
{
}

void SplitEvent::setPos(float val)
{
	m_pos = val;
}

void SplitEvent::setAxis(int val)
{
	m_axis = val;
}
	
float SplitEvent::getPos() const
{
	return m_pos;
}

int SplitEvent::getAxis() const
{
	return m_axis;
}
