/*
 *  RayIntersectionContext.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "RayIntersectionContext.h"

RayIntersectionContext::RayIntersectionContext()
{
	reset();
}

RayIntersectionContext::~RayIntersectionContext() {}

void RayIntersectionContext::reset()
{
	m_level = 0;
	m_minHitDistance = 10e10;
	m_success = 0;
	m_cell = 0;
}

void RayIntersectionContext::setBBox(const BoundingBox & bbox)
{
	m_bbox = bbox;
}

BoundingBox RayIntersectionContext::getBBox() const
{
	return m_bbox;
}

void RayIntersectionContext::verbose() const
{
	std::cout<<" bbox "<<m_bbox.getMin(0)<<" "<<m_bbox.getMin(1)<<" "<<m_bbox.getMin(2)<<" - "<<m_bbox.getMax(0)<<" "<<m_bbox.getMax(1)<<" "<<m_bbox.getMax(2)<<"\n";
}