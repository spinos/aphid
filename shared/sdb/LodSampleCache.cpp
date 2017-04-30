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

const SampleCache::ASample * SampleCache::data() const
{ return m_data.get(); }

const SampleCache::ASample & SampleCache::getASample(const int & i) const
{ return m_data[i]; }

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

void SampleCache::setColorByUV()
{
	for(int i=0;i<m_numSamples;++i) {
		ASample & d = m_data[i];
		d.col.set(d.u, d.v, 0.f);
	}
}

LodSampleCache::LodSampleCache(Entity * parent) : LodGrid(parent)
{}

LodSampleCache::~LodSampleCache()
{}

bool LodSampleCache::hasLevel(int x)
{ return m_samples[x].numSamples() > 0; }

void LodSampleCache::buildSampleCache(int minLevel, int maxLevel)
{	
	for(int i=minLevel;i<=maxLevel;++i) {
		SampleCache & spsi = m_samples[i];
		spsi.clear();
		int ns = countLevelNodes(i);
		if(ns < 1) {
			continue;
		}
		spsi.create(ns);
		dumpLevelSamples<SampleCache::ASample>(spsi.data(), i );
		spsi.assignNoise();
		//spsi.setColorByUV();
		
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

void LodSampleCache::clear()
{
	for(int i=0;i<8;++i) {
		m_samples[i].clear();
	}
	LodGrid::clear();
}

void LodSampleCache::reshuffleAtLevel(const int & level)
{
	m_samples[level].assignNoise();
}

}

}