/*
 *  SampleFilter.cpp
 *  
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SampleFilter.h"
#include <img/ExrImage.h>

namespace aphid {

SampleFilter::SampleFilter()
{ 
    m_imageSampler = NULL;
    m_portion = .8f; 
}

SampleFilter::~SampleFilter()
{}

void SampleFilter::setPortion(const float & x)
{ 
	m_portion = x; 
	if(m_portion < .05f) {
		m_portion = .05f;
	}
}

void SampleFilter::setMode(SelectionContext::SelectMode mode)
{
	m_mode = mode;
}

bool SampleFilter::isRemoving() const
{ return m_mode == SelectionContext::Remove; }

bool SampleFilter::isReplacing() const
{ return m_mode == SelectionContext::Replace; }

bool SampleFilter::isAppending() const
{ return m_mode == SelectionContext::Append; }

const int & SampleFilter::maxSampleLevel() const
{ return m_maxSampleLevel; }

const float & SampleFilter::sampleGridSize() const
{ return m_sampleGridSize; }

void SampleFilter::computeGridLevelSize(const float & cellSize,
				const float & sampleDistance)
{
	m_sampleGridSize = sampleDistance * 2.f;
	m_maxSampleLevel = 0;
	for(;m_maxSampleLevel < 6;++m_maxSampleLevel) {
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

bool SampleFilter::throughPortion(const float & x) const
{ return x < m_portion; }

const float & SampleFilter::portion() const
{ return m_portion; }

bool SampleFilter::throughNoise3D(const Vector3F & p) const
{
	if(m_noiseLevel < 1e-2f) {
		return true;
	}
	
	return sampleNoise3((const float *)&p) > m_noiseLevel;
}

bool SampleFilter::throughImage(const float & k, const float & s, const float & t) const
{
    if(!m_imageSampler) {
        return true;
    }
    float texCol[3];
    m_imageSampler->sample(s, t, 1, texCol);			
    return k < texCol[0];
}

}