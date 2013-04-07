/*
 *  TargetGraph.cpp
 *  hc
 *
 *  Created by jian zhang on 4/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TargetGraph.h"

TargetGraph::TargetGraph() {}
TargetGraph::~TargetGraph() {}

void TargetGraph::createTargetIndices(unsigned num)
{
	m_targetIndices = new unsigned[num];
}

void TargetGraph::setTargetTriangle(unsigned idx, unsigned a, unsigned b, unsigned c)
{
	unsigned *i = m_targetIndices;
	i += idx * 3;
	*i = a;
	i++;
	*i = b;
	i++;
	*i = c;
}

void TargetGraph::setControlId(unsigned idx)
{
	m_controlId = idx;
}
