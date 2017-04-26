/*
 *  VegetationPatch.cpp
 *  garden
 *
 *  Created by jian zhang on 4/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "VegetationPatch.h"
#include "PlantPiece.h"
#include <math/miscfuncs.h>

using namespace aphid;

VegetationPatch::VegetationPatch() :
m_yardR(.5f),
m_tilt(0.f)
{
	m_translatev[0] = 0.f;
	m_translatev[1] = 0.f;
	m_translatev[2] = 0.f;
}

VegetationPatch::~VegetationPatch()
{
	clearPlants();
}

int VegetationPatch::numPlants() const
{ return m_plants.size(); }

const PlantPiece * VegetationPatch::plant(const int & i) const
{ return m_plants[i]; }

bool VegetationPatch::addPlant(PlantPiece * pl)
{
	const float r = pl->exclR();
	const float px = RandomFn11() * m_yardR;
	const float pz = RandomFn11() * m_yardR;
	Vector3F pos(px, 0.f, pz);
	
	if(pos.length() > m_yardR) {
		return false;
	}
	
	Matrix44F rotm;
	rotm.rotateX(m_tilt);
	pos = rotm.transform(pos);
	
	const float sc = RandomF01() * -.19f + 1.f;
	
	if(intersectPlants(pos, r * sc) ) {
		return false;
	}
	
	Matrix44F tm;
	tm.scaleBy(sc);
	const float ry = RandomFn11() * PIF;
	tm.rotateY(ry);
	tm.setTranslation(pos);
	
	pl->setTransformMatrix(tm);
	
	m_plants.push_back(pl);
	
	float nr = r * 2.f + pos.length();
	if(nr > r * 14.f) {
		nr = r * 14.f;
	}
	if(m_yardR < nr) {
		m_yardR = nr;
	}
	
	std::cout.flush();
	
	return true;
}

bool VegetationPatch::intersectPlants(const aphid::Vector3F & pos, const float & r) const
{
	PlantListTyp::const_iterator it = m_plants.begin();
	for(;it!=m_plants.end();++it) {
		const float & ar = (*it)->exclR();
		const Matrix44F & amat = (*it)->transformMatrix();
		const float sx = amat.scale().x;
		
		if(pos.distanceTo(amat.getTranslation() ) < (r + ar * sx + .1f)) {
			return true;
		}
	}
	return false;
}

void VegetationPatch::clearPlants()
{
	PlantListTyp::iterator it = m_plants.begin();
	for(;it!=m_plants.end();++it) {
		delete *it;
	}
	m_plants.clear();
}

bool VegetationPatch::isFull() const
{
	return numPlants() > 99;
}

void VegetationPatch::setTilt(const float & x)
{ m_tilt = x; }

const float & VegetationPatch::tilt() const
{ return m_tilt; }

const float & VegetationPatch::yardRadius() const
{ return m_yardR; }

void VegetationPatch::setTranslation(const float & px,
		const float & py,
		const float & pz)
{
	m_translatev[0] = px;
	m_translatev[1] = py;
	m_translatev[2] = pz;
}

const float * VegetationPatch::translationV() const
{ return &m_translatev[0]; }
