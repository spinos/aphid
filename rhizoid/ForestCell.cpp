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

}
