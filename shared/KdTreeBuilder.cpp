/*
 *  KdTreeBuilder.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "KdTreeBuilder.h"

KdTreeBuilder::KdTreeBuilder() 
{
}

KdTreeBuilder::~KdTreeBuilder() {}

void KdTreeBuilder::calculateSplitEvents(const PartitionBound &bound)
{
	int axis = bound.bbox.getLongestAxis();
	m_event.setAxis(axis);
	m_event.setPos(bound.bbox.getMin(axis) * 0.5f + bound.bbox.getMax(axis) * 0.5f);
}

SplitEvent &KdTreeBuilder::bestSplit()
{
	return m_event;
}
