/*
 *  MinMaxBins.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "MinMaxBins.h"
#include <iostream>
MinMaxBins::MinMaxBins() : m_isFlat(0), m_minBin(NULL), m_maxBin(NULL) {}
MinMaxBins::~MinMaxBins() 
{
	if(m_minBin) delete[] m_minBin;
	if(m_maxBin) delete[] m_maxBin;
}
	
void MinMaxBins::create(const unsigned &num, const float &min, const float &max)
{
	m_binSize = num;
	m_boundLeft = min; 
	m_delta = (max - min) / num;
	m_minBin = new unsigned[num];
	m_maxBin = new unsigned[num];
	
	for(unsigned i = 0; i < m_binSize; i++) {
		m_minBin[i] = m_maxBin[i] = 0;
	}
}

void MinMaxBins::add(const float &min, const float &max)
{
	int minIdx = (min - m_boundLeft) / m_delta;
	validateIdx(minIdx);
	m_minBin[minIdx]++;
	
	int maxIdx = (max - m_boundLeft) / m_delta;
	validateIdx(maxIdx);
	m_maxBin[maxIdx]++;
}

void MinMaxBins::scan()
{
	for(unsigned i = 1; i < m_binSize; i++)
		m_minBin[i] += m_minBin[i - 1];
	for(int i = m_binSize - 2; i >= 0; i--)
		m_maxBin[i] += m_maxBin[i + 1];
}

void MinMaxBins::get(const unsigned &idx, unsigned &left, unsigned &right) const
{
	left = m_minBin[idx];
	right =	m_maxBin[idx + 1];
}

void MinMaxBins::validateIdx(int &idx) const
{
	if(idx < 0) {
	    //printf("too low %i ", idx);	
	    idx = 0;
	}
	else if(idx > ((int)m_binSize - 1)) {
	    //printf("too high %i ", idx);	
	    idx = m_binSize - 1;
	}
}

char MinMaxBins::isFlat() const
{
	return m_isFlat;
}

void MinMaxBins::setFlat()
{
	m_isFlat = 1;
}
