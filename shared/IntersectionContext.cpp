/*
 *  IntersectionContext.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "IntersectionContext.h"

IntersectionContext::IntersectionContext()
{
	reset();
}

IntersectionContext::~IntersectionContext() {}

void IntersectionContext::reset()
{
	m_level = 0;
	m_minHitDistance = 10e10;
	m_success = 0;
	m_cell = 0;
}

void IntersectionContext::setBBox(const BoundingBox & bbox)
{
	m_bbox = bbox;
}

BoundingBox IntersectionContext::getBBox() const
{
	return m_bbox;
}

void IntersectionContext::verbose() const
{
	std::cout<<" bbox "<<m_bbox.getMin(0)<<" "<<m_bbox.getMin(1)<<" "<<m_bbox.getMin(2)<<" - "<<m_bbox.getMax(0)<<" "<<m_bbox.getMax(1)<<" "<<m_bbox.getMax(2)<<"\n";
}