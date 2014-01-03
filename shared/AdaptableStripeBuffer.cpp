/*
 *  AdaptableStripeBuffer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/2/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptableStripeBuffer.h"

AdaptableStripeBuffer::AdaptableStripeBuffer()
{
	m_numCvs  = 0;
	m_pos = 0;
	m_col = 0;
	m_width = 0;
	m_maxNumStripe = 0;
	m_maxNumCv = 0;
	m_useNumStripe = 0;
	m_currentStripe = 0;
}

AdaptableStripeBuffer::~AdaptableStripeBuffer()
{
	clear();
}

void AdaptableStripeBuffer::clear()
{
	if(m_numCvs) delete[] m_numCvs;
	if(m_pos) delete[] m_pos;
	if(m_col) delete[] m_col;
	if(m_width) delete[] m_width;
	m_numCvs  = 0;
	m_pos = 0;
	m_col = 0;
	m_width = 0;
}

void AdaptableStripeBuffer::create(unsigned maxNumStripe, unsigned numCvPerStripe)
{
	if(maxNumStripe <= m_maxNumStripe && maxNumStripe * numCvPerStripe <= m_maxNumCv) return;
	clear();
	m_maxNumStripe = maxNumStripe;
	m_maxNumCv = maxNumStripe * numCvPerStripe;
	
	m_numCvs = new unsigned[m_maxNumStripe];
	m_pos = new Vector3F[m_maxNumCv];
	m_col = new Vector3F[m_maxNumCv];
	m_width = new float[m_maxNumCv];
}

unsigned AdaptableStripeBuffer::numStripe() const
{
	return m_useNumStripe;
}

unsigned AdaptableStripeBuffer::numPoints() const
{
	return m_numCvs[m_useNumStripe - 1] + m_currentStripe;
}

void AdaptableStripeBuffer::begin()
{
	m_useNumStripe = 0;
	m_currentStripe = 0;
}

void AdaptableStripeBuffer::next()
{
	m_currentStripe += m_numCvs[m_useNumStripe];
	m_useNumStripe++;
}

char AdaptableStripeBuffer::end() const
{
	return m_useNumStripe == m_maxNumStripe;
}

unsigned * AdaptableStripeBuffer::numCvs()
{
	return m_numCvs;
}

Vector3F * AdaptableStripeBuffer::pos()
{
	return m_pos;
}

Vector3F * AdaptableStripeBuffer::col()
{
	return m_col;
}

float * AdaptableStripeBuffer::width()
{
	return m_width;
}

unsigned * AdaptableStripeBuffer::currentNumCvs()
{
	return &m_numCvs[m_useNumStripe];
}

Vector3F * AdaptableStripeBuffer::currentPos()
{
	return &m_pos[m_currentStripe];
}

Vector3F * AdaptableStripeBuffer::currentCol()
{
	return &m_col[m_currentStripe];
}

float * AdaptableStripeBuffer::currentWidth()
{
	return &m_width[m_currentStripe];
}
