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
#include <sdb/VectorArray.h>
#include <geom/ConvexShape.h>
#include <math/Matrix44F.h>
#include <math/miscfuncs.h>
#include <sdb/ValGrid.h>

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

void VegetationPatch::addPlant(PlantPiece * pl)
{
	m_plants.push_back(pl);
	float nr = pl->exclR() * 12.f;
	if(m_yardR < nr) {
		m_yardR = nr;
	}
}

bool VegetationPatch::intersectPlants(const Vector3F & pos, const float & r) const
{
	PlantListTyp::const_iterator it = m_plants.begin();
	for(;it!=m_plants.end();++it) {
		const float & ar = (*it)->exclR();
		const Matrix44F & amat = (*it)->transformMatrix();
		const float sx = amat.scale().x;
		
		if(pos.distanceTo(amat.getTranslation() ) < (r + ar * sx)) {
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
	m_yardR = .5f;
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

void VegetationPatch::extractGeomIds(int * dst)
{
	int it = 0;
	const int n = numPlants();
	for(int i=0;i<n;++i) {
		m_plants[i]->extractGeomIds(dst, it);
	}
}

void VegetationPatch::getGeom(GeomElmArrTyp * dst,
					BoundingBox & box)
{
	const Matrix44F space;
	const int n = numPlants();
	for(int i=0;i<n;++i) {
		m_plants[i]->getGeom(dst, box, space);
	}
}

void VegetationPatch::setTriangleDrawCache(const GeomElmArrTyp & src)
{
	const int n = src.size();
	setTriDrawBufLen(n * 3);
	Vector3F * pos = triPositionR();
	Vector3F * nml = triNormalR();
	Vector3F * col = triColorR();
	for(int i=0; i<n; ++i) {
		const GeomElmTyp * atri = src.get(i);
		pos[i*3] = atri->P(0);
		pos[i*3+1] = atri->P(1);
		pos[i*3+2] = atri->P(2);
		
		nml[i*3] = atri->N(0);
		nml[i*3+1] = atri->N(1);
		nml[i*3+2] = atri->N(2);
		
		col[i*3] = atri->C(0);
		col[i*3+1] = atri->C(1);
		col[i*3+2] = atri->C(2);
	}
}

void VegetationPatch::voxelize3(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	ExampVox::voxelize3(tri, bbox);
	setGeomBox2(bbox);
	ExampVox::buildVoxel(bbox);
	
}
