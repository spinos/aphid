/*
 *  SynthesisGroup.cpp
 * 
 *  instance geom_ind and tm_mat
 *
 *  Created by jian zhang on 8/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SynthesisGroup.h"
#include <math/Matrix44F.h>

namespace gar {

SynthesisGroup::SynthesisGroup() :
m_numInstances(0)
{}

SynthesisGroup::~SynthesisGroup()
{
	m_geoms.clear();
	m_tms.clear();
}

const int& SynthesisGroup::numInstances() const
{ return m_numInstances; }

void SynthesisGroup::addInstance(const int& geom, const aphid::Matrix44F& tm)
{ 
	m_geoms.push_back(geom);
	m_tms.push_back(tm);
	m_numInstances++;
}

void SynthesisGroup::getInstance(int& geom, aphid::Matrix44F& tm, const int& i)
{
	geom = m_geoms[i];
	tm = m_tms[i];
}

}