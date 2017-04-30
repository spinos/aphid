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
#include <math/Matrix44F.h>
#include <math/miscfuncs.h>

using namespace aphid;

VegetationPatch::VegetationPatch() :
m_yardR(.5f),
m_tilt(0.f)
{
	Matrix44F eye;
	eye.glMatrix(m_tmv);
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
	const Vector3F pos(px, 0.f, pz);
	
	if(pos.length() > m_yardR) {
		return false;
	}
	
	const float sc = RandomF01() * -.19f + 1.f;
	
	if(intersectPlants(pos, r * sc) ) {
		return false;
	}
	
	Matrix44F tm;
	tm.scaleBy(sc);
	const float ry = RandomFn11() * PIF;
	tm.rotateY(ry);
	tm.rotateX(-m_tilt);
	tm.setTranslation(pos);
	
	pl->setTransformMatrix(tm);
	
	m_plants.push_back(pl);
	
	float nr = r * 2.f + pos.length();
	if(nr > r * 12.f) {
		nr = r * 12.f;
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
	return numPlants() > 79;
}

void VegetationPatch::setTilt(const float & x)
{ m_tilt = x; }

const float & VegetationPatch::tilt() const
{ return m_tilt; }

const float & VegetationPatch::yardRadius() const
{ return m_yardR; }

void VegetationPatch::setTransformation(const Matrix44F & tm)
{ tm.glMatrix(m_tmv); }

const float * VegetationPatch::transformationV() const
{ return &m_tmv[0]; }

int VegetationPatch::getNumTms()
{
	int cc = 0;
	const int n = numPlants();
	for(int i=0;i<n;++i) {
		m_plants[i]->countNumTms(cc);
	}
	return cc;
}

void VegetationPatch::extractTms(Matrix44F * dst)
{
	int it = 0;
	const int n = numPlants();
	for(int i=0;i<n;++i) {
		m_plants[i]->extractTms(dst, it);
	}
}
