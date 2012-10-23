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

KdTreeBuilder::~KdTreeBuilder() 
{
	//printf("builder quit\n");
}

void KdTreeBuilder::calculateSplitEvents(const PartitionBound &bound)
{
	int axis = bound.bbox.getLongestAxis();
	m_event.setAxis(axis);
	m_event.setPos(bound.bbox.getMin(axis) * 0.5f + bound.bbox.getMax(axis) * 0.5f);
	m_event.calculateSides(bound);
}

const SplitEvent *KdTreeBuilder::bestSplit() const
{
	return &m_event;
}
