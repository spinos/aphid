/*
 *  SampleFilter.cpp
 *  
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SampleFilter.h"

namespace aphid {

SampleFilter::SampleFilter()
{}

SampleFilter::~SampleFilter()
{}

void SampleFilter::setMode(SelectionContext::SelectMode mode)
{
	m_mode = mode;
}

void SampleFilter::setSphere(const Vector3F & center,
						const float & radius)
{
	m_center = center;
	m_radius = radius;
}

bool SampleFilter::insideSphere(const Vector3F & p) const
{
	return p.distanceTo(m_center) < m_radius;
}

Vector3F SampleFilter::boxLow() const
{ 
	return Vector3F(m_center.x - m_radius,
						m_center.y - m_radius,
						m_center.z - m_radius); 
}
	
Vector3F SampleFilter::boxHigh() const
{ 
	return Vector3F(m_center.x + m_radius,
						m_center.y + m_radius,
						m_center.z + m_radius); 
}

bool SampleFilter::isRemoving() const
{ return m_mode == SelectionContext::Remove; }

}