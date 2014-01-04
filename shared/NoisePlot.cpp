/*
 *  NoisePlot.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "NoisePlot.h"
#include <PseudoNoise.h>
NoisePlot::NoisePlot() {m_samps = 0;}
NoisePlot::~NoisePlot() {clear();}
void NoisePlot::clear() {
	if(m_samps) delete[] m_samps;
	m_samps = 0;
}

void NoisePlot::createPlot(unsigned n)
{
	clear();
	m_numSamps = 64;
	while(m_numSamps < n) m_numSamps *= 2;
	m_samps = new float[m_numSamps+1];
}

void NoisePlot::computePlot(unsigned seed)
{
	PseudoNoise noi;
	for(unsigned i = 0; i <= m_numSamps; i++) 
		m_samps[i] = noi.rfloat(seed + i * 17) - .5f;
}

float NoisePlot::getNoise(float u, float lod) const
{
	if(lod == 1.f) return getFull(u);
	
	const unsigned ng = getGridSize(lod);
	float hi = getLevel(u, ng);
	float portion = lod * 5 - (int)(lod * 5);
	if(portion == 0.f) return hi;
	float lo = getLevel(u, ng * 2);
	return hi * portion + lo * (1.f - portion);
}

float NoisePlot::getFull(float u) const
{
	float portion = u * m_numSamps;
	int i = portion;
	portion -= i;
	return m_samps[i] * (1.f - portion) + m_samps[i+1] * portion;
}

unsigned NoisePlot::getGridSize(float lod) const
{
	unsigned r = m_numSamps / 16;
	for(int i = 0; i < lod * 5; i++) {
		r /= 2;
	}
	return r;
}

float NoisePlot::getLevel(float u, unsigned ng) const
{
	float coord = u * m_numSamps;
	float portion = coord /(float)ng;
	int i = portion;
	portion -= i;
	return m_samps[i*ng] * (1.f - portion) + m_samps[(i+1)*ng] * portion;
}
