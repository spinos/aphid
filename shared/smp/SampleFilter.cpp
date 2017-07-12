/*
 *  SampleFilter.cpp
 *  
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SampleFilter.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace smp {

SampleFilter::SampleFilter() :
m_facing(Vector3F(0.f, 1.f, 0.f) ),
m_portion(1.f),
m_angle(-1.f),
m_numFilteredSamples(0)
{}

SampleFilter::~SampleFilter()
{}

const int& SampleFilter::numFilteredSamples() const
{ return m_numFilteredSamples; }

const Vector3F* SampleFilter::filteredSamples() const
{ return m_samples.get(); }

void SampleFilter::setPortion(const float& x)
{ m_portion = x; }

void SampleFilter::setFacing(const Vector3F& v)
{ m_facing = v.normal(); }

void SampleFilter::setAngle(const float& x)
{ m_angle = x; }

bool SampleFilter::isFiltered(const Vector3F& v) const
{
	if(m_portion < 1.f) {
		if(RandomF01() > m_portion) {
			return true;
		}
	}
	
	if(m_angle > -1.f) {
		
		const Vector3F nv = v.normal();
		if(nv.dot(m_facing) < m_angle ) {
			return true;
		}
	}
	
	return false; 
}

}

}
