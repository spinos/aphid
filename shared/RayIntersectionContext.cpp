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
	m_componentType = TFace;
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

void RayIntersectionContext::setComponentFilerType(ComponentType ft)
{
	m_componentType = ft;
}

void RayIntersectionContext::verbose() const
{
	std::cout<<" bbox "<<m_bbox.min(0)<<" "<<m_bbox.min(1)<<" "<<m_bbox.min(2)<<" - "<<m_bbox.max(0)<<" "<<m_bbox.max(1)<<" "<<m_bbox.max(2)<<"\n";
}