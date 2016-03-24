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
namespace aphid {

IntersectionContext::IntersectionContext() {}

IntersectionContext::~IntersectionContext() {}

void IntersectionContext::reset()
{
	m_level = 0;
	m_minHitDistance = 10e10;
	m_success = 0;
	m_cell = 0;
	m_enableNormalRef = 0;
	twoSided = 0;
}

void IntersectionContext::reset(const Ray & ray)
{
    m_ray = ray;
	reset();
}

void IntersectionContext::setBBox(const BoundingBox & bbox)
{
	m_bbox = bbox;
}

BoundingBox IntersectionContext::getBBox() const
{
	return m_bbox;
}

void IntersectionContext::setNormalReference(const Vector3F & nor)
{
	m_refN = nor;
	m_enableNormalRef = 1;
}

void IntersectionContext::verbose() const
{
	std::cout<<" bbox "<<m_bbox.getMin(0)<<" "<<m_bbox.getMin(1)<<" "<<m_bbox.getMin(2)<<" - "<<m_bbox.getMax(0)<<" "<<m_bbox.getMax(1)<<" "<<m_bbox.getMax(2)<<"\n";
}

BoxIntersectContext::BoxIntersectContext() :
m_cap(1),
m_exact(false)
{}

BoxIntersectContext::~BoxIntersectContext()
{}

void BoxIntersectContext::reset(int maxNumPrim, bool beExact)
{ 
	m_prims.clear(); 
	m_cap = maxNumPrim;
	m_exact = beExact;
}

void BoxIntersectContext::addPrim(const int & i)
{ m_prims.push_back(i); }

int BoxIntersectContext::numIntersect() const
{ return m_prims.size(); }

bool BoxIntersectContext::isExact() const
{ return m_exact; }

bool BoxIntersectContext::isFull() const
{ return m_prims.size() >= m_cap; }

}