/*
 *  ForestCell.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ForestCell.h"
#include <PlantCommon.h>

namespace aphid {

ForestCell::ForestCell(Entity * parent) : sdb::Array<sdb::Coord2, Plant>(parent)
{
	m_lodsamp = new sdb::LodSampleCache;
	m_numActiveSamples = 0;
}

ForestCell::~ForestCell()
{
	delete m_lodsamp;
}

const sdb::SampleCache * ForestCell::sampleAtLevel(int level) const
{ return m_lodsamp->samplesAtLevel(level); }

const float * ForestCell::samplePoints(int level) const
{ return sampleAtLevel(level)->points(); }

const float * ForestCell::sampleNormals(int level) const
{ return sampleAtLevel(level)->normals(); }

const int & ForestCell::numSamples(int level) const
{ return m_lodsamp->numSamplesAtLevel(level); }

bool ForestCell::hasSamples(int level) const
{ return m_lodsamp->hasLevel(level); }

void ForestCell::deselectSamples()
{
	m_activeSampleKeys.clear();
	m_numActiveSamples = 0;
}

const int & ForestCell::numSelectedSamples() const
{ return m_numActiveSamples; }

void ForestCell::updateActiveIndices()
{
	m_numActiveSamples = m_activeSampleKeys.size();
	if(m_numActiveSamples < 1) {
		return;
	}
	
	int c=0;
	m_activeSampleKeys.begin();
	while(!m_activeSampleKeys.end() ) {
	
		m_activeSampleIndices[c] = m_activeSampleKeys.key();
		c++;
		
		m_activeSampleKeys.next();
	}
}

const int * ForestCell::selectedSampleIndices() const
{ return m_activeSampleIndices.get(); }

void ForestCell::clearSamples()
{
	m_lodsamp->clear();
}

}
