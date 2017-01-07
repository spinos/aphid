/*
 *  FeatherGeomParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherGeomParam.h"
#include "Geom1LineParam.h"

using namespace aphid;

FeatherGeomParam::FeatherGeomParam() 
{
	m_lines[0] = new Geom1LineParam(4);
	for(int i=1;i<5;++i) {
		m_lines[i] = new Geom1LineParam(5);
	}
	m_lines[1]->setRotateOffsetZ(0.025f);
	m_lines[2]->setRotateOffsetZ(0.05f);
	m_lines[3]->setRotateOffsetZ(-0.025f);
	m_lines[4]->setRotateOffsetZ(-0.05f);
	
}

FeatherGeomParam::~FeatherGeomParam()
{
	for(int i=0;i<5;++i) {
		delete m_lines[i];
	}
}

void FeatherGeomParam::setFlying(const int * nps,
						const float * chords,
						const float * ts)
{
	m_lines[0]->set(nps, chords, ts);
}

void FeatherGeomParam::setCovert(int i,
						const int * nps,
						const float * chords,
						const float * ts)
{
	m_lines[i]->set(nps, chords, ts);
}

bool FeatherGeomParam::isChanged() const
{
	for(int i=0;i<5;++i) {
		if(m_lines[i]->isChanged() ) {
			return true;
		}
	}
	return false;
}

const float & FeatherGeomParam::longestChord() const
{ return m_lines[0]->longestChord(); }

Geom1LineParam * FeatherGeomParam::line(int i)
{ return m_lines[i]; }
