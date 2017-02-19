/*
 *  GrowOption.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GrowOption.h"
#include <img/ExrImage.h>

namespace aphid {

GrowOption::GrowOption() 
{
	m_upDirection = Vector3F::YAxis;
	m_alongNormal = 0;
	m_minScale = 1.f;
	m_maxScale = 1.f;
	m_rotateNoise = 0.f;
	m_plantId = 0;
	m_multiGrow = 1;
	m_minMarginSize = .1f;
	m_maxMarginSize = .1f;
	m_strength = .67f;
	m_stickToGround = true;
	m_sampler = NULL;
}

GrowOption::~GrowOption()
{
	if(m_sampler) {
		delete m_sampler;
	}
}

void GrowOption::setStrokeMagnitude(const float & x) 
{
	m_strokeMagnitude = x;
	if(m_strokeMagnitude < -.5f) {
		m_strokeMagnitude = -.5f;
	}
	if(m_strokeMagnitude > .5f) {
		m_strokeMagnitude = .5f;
	}
}

bool GrowOption::openImage(const std::string & fileName)
{
	if(!m_sampler) {
		m_sampler = new ExrImage;
	}
	
	bool stat = m_sampler->read(fileName);
	m_sampler->verbose();
	
	return stat;
}

void GrowOption::closeImage()
{
	if(m_sampler) {
		delete m_sampler;
		m_sampler = NULL;
	}
}

bool GrowOption::hasSampler() const
{
	if(!m_sampler) {
		return false;
	}
	return m_sampler->isValid();
	
}

std::string GrowOption::imageName() const
{ return m_sampler->fileName(); }

void GrowOption::sampleRed(float * col, const float & u,
					const float & v) const
{ m_sampler->sample(u, v, 1, col); }
		
}