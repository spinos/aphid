/*
 *  LodSampleCache.cpp
 *  
 *
 *  Created by jian zhang on 2/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "LodSampleCache.h"

namespace aphid {

namespace sdb {

const int SampleCache::DataStride = 32;

SampleCache::SampleCache()
{ m_numSamples = 0; }

SampleCache::~SampleCache()
{}

void SampleCache::create(int n)
{ 
	m_data.reset(new ASample[n]);
	m_numSamples = n; 
}

void SampleCache::clear()
{
	m_data.reset();
	m_numSamples = 0; 
}

const int & SampleCache::numSamples() const
{ return m_numSamples; }

SampleCache::ASample * SampleCache::data()
{ return m_data.get(); }

const float * SampleCache::points() const
{ return (const float *)&m_data[0]; }

const float * SampleCache::normals() const
{ return (const float *)&(m_data[0].nml); }

LodSampleCache::LodSampleCache(Entity * parent) : LodGrid(parent)
{}

LodSampleCache::~LodSampleCache()
{}

bool LodSampleCache::hasLevel(int x)
{ return m_samples[x].numSamples() > 0; }

void LodSampleCache::buildSamples(int minLevel, int maxLevel)
{	
	for(int i=minLevel;i<=maxLevel;++i) {
		SampleCache & spsi = m_samples[i];
		spsi.clear();
		int ns = countLevelNodes(i);
		if(ns>0) {
			spsi.create(ns);
			dumpLevelSamples<SampleCache::ASample>(spsi.data(), i );
		}
	}
}

const int & LodSampleCache::numSamplesAtLevel(int x) const
{
	return m_samples[x].numSamples();
}

const SampleCache * LodSampleCache::samplesAtLevel(int x) const
{ return &m_samples[x]; }

}

}