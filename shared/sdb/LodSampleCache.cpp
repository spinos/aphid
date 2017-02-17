/*
 *  LodSampleCache.cpp
 *  
 *
 *  Created by jian zhang on 2/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "LodSampleCache.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace sdb {

const int SampleCache::DataStride = 64;

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

const float * SampleCache::colors() const
{ return (const float *)&(m_data[0].col); }

void SampleCache::assignNoise()
{
	for(int i=0;i<m_numSamples;++i) {
		m_data[i].noi = RandomF01();
	}
}

void SampleCache::setColorByNoise()
{
	for(int i=0;i<m_numSamples;++i) {
		ASample & d = m_data[i];
		d.col.set(d.noi, d.noi, d.noi);
	}
}

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
			spsi.assignNoise();
		}
	}
}

const int & LodSampleCache::numSamplesAtLevel(int x) const
{
	return m_samples[x].numSamples();
}

const SampleCache * LodSampleCache::samplesAtLevel(int x) const
{ return &m_samples[x]; }

SampleCache * LodSampleCache::samplesAtLevel(int x)
{ return &m_samples[x]; }

}

}