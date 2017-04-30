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

float NoisePlot::sample(unsigned idx) const
{
	return m_samps[idx % m_numSamps];
}

unsigned NoisePlot::numSamples() const
{
	return m_numSamps;
}