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
	m_bbox.setMin(m_center.x - m_radius,
						m_center.y - m_radius,
						m_center.z - m_radius);
	m_bbox.setMax(m_center.x + m_radius,
						m_center.y + m_radius,
						m_center.z + m_radius);
}

void SampleFilter::limitBox(const BoundingBox & b)
{ 
	m_bbox.shrinkBy(b); 
}

Vector3F SampleFilter::boxLow() const
{ 
	return m_bbox.lowCorner(); 
}

Vector3F SampleFilter::boxHigh() const
{ 
	return m_bbox.highCorner(); 
}

bool SampleFilter::isRemoving() const
{ return m_mode == SelectionContext::Remove; }

bool SampleFilter::isReplacing() const
{ return m_mode == SelectionContext::Replace; }

bool SampleFilter::isAppending() const
{ return m_mode == SelectionContext::Append; }

bool SampleFilter::intersect(const BoundingBox & b) const
{ return m_bbox.intersect(b); }

bool SampleFilter::intersect(const Vector3F & p) const
{ 
	if(!m_bbox.isPointInside(p) ) {
		return false;
	}
	
	return p.distanceTo(m_center) < m_radius; 
}

const int & SampleFilter::maxSampleLevel() const
{ return m_maxSampleLevel; }

const float & SampleFilter::sampleGridSize() const
{ return m_sampleGridSize; }

void SampleFilter::computeGridLevelSize(const float & cellSize,
				const float & sampleDistance)
{
	m_sampleGridSize = sampleDistance * 2.3f;
	m_maxSampleLevel = 0;
	for(;m_maxSampleLevel < 5;++m_maxSampleLevel) {
		if(m_sampleGridSize > cellSize) {
			break;
		}
		m_sampleGridSize *= 2.f;
	}
	if(m_sampleGridSize < cellSize * .5f) {
		m_sampleGridSize = cellSize * .5f;
	}
	else if(m_sampleGridSize > cellSize) {
		m_sampleGridSize = cellSize;
	}
}

}