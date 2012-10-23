/*
 *  SplitCandidate.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "SplitCandidate.h"
int SplitCandidate::Dimension = 3;
SplitCandidate::SplitCandidate() {}

void SplitCandidate::setPos(float val)
{
	m_pos = val;
}

void SplitCandidate::setAxis(int val)
{
	m_axis = val;
}
	
float SplitCandidate::getPos() const
{
	return m_pos;
}

int SplitCandidate::getAxis() const
{
	return m_axis;
}

void SplitCandidate::verbose() const
{
	printf("split at %f ", m_pos);
	if(m_axis == 0)
		printf("along X axis\n");
	else if(m_axis == 1)
		printf("along Y axis\n");
	else
		printf("along Z axis\n");
}