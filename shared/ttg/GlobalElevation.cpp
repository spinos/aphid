/*
 *  GlobalElevation.cpp
 *  
 *
 *  Created by jian zhang on 3/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GlobalElevation.h"
#include <img/HeightField.h>
#include <img/ExrImage.h>

namespace aphid {

namespace ttg {

GlobalElevation::GlobalElevation()
{
	m_planetCenter.set(0.f, -3.3895e6f, 0.f);/// mean radius of Mars
}

GlobalElevation::~GlobalElevation()
{
	internalClear();
}

void GlobalElevation::setPlanetRadius(float x)
{ m_planetCenter.y = -x; }

float GlobalElevation::sample(const Vector3F & pos) const
{
	return pos.distanceTo(m_planetCenter) + m_planetCenter.y;
}

int GlobalElevation::numHeightFiles() const
{ return m_fields.size(); }

void GlobalElevation::internalClear()
{
	FieldVecTyp::iterator it = m_fields.begin();
	for(;it!=m_fields.end();++it) {
		delete *it;
	}
	m_fields.clear();
}

bool GlobalElevation::loadHeightField(const std::string & fileName)
{
	ExrImage exr;
	exr.read(fileName);
	exr.verbose();
	if(!exr.isValid() ) {
		return false;
	}
	
	Array3<float> inputX;
	exr.sampleRed(inputX);
	
	img::HeightField * afld = new img::HeightField(inputX);
	m_fields.push_back(afld);
	
	afld->calculateBBox();
	
	return true;
}

}

}