/*
 *  Vegetation.cpp
 *  
 *
 *  Created by jian zhang on 4/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Vegetation.h"
#include "VegetationPatch.h"
#include <geom/ATriangleMesh.h>
#include <sdb/VectorArray.h>
#include <geom/ConvexShape.h>
#include "gar_common.h"
#include "data/grass.h"
#include <boost/format.hpp>
#include <cmath>

using namespace aphid;

Vegetation::Vegetation() :
m_numPatches(1)
{
	for(int i=0;i<TOTAL_NUM_P;++i) {
		m_patches[i] = new VegetationPatch;
	}
	const float deltaAng = .8f / ((float)NUM_ANGLE - 1);
	for(int j=0;j<NUM_ANGLE;++j) {
		const float angj = deltaAng * j;
		for(int i=0;i<NUM_VARIA;++i) {
			m_patches[j * NUM_VARIA + i]->setTilt(angj);
		}
	}
}

Vegetation::~Vegetation()
{
	for(int i=0;i<TOTAL_NUM_P;++i) {
		delete m_patches[i];
	}
	clearCachedGeom();
}

VegetationPatch * Vegetation::patch(const int & i)
{
	return m_patches[i];
}

void Vegetation::setNumPatches(int x)
{ m_numPatches = x; }

const int & Vegetation::numPatches() const
{ return m_numPatches; }

int Vegetation::getMaxNumPatches() const
{ return TOTAL_NUM_P; }

void Vegetation::rearrange()
{
	Matrix44F tm;
	float px, pz = 0.f, py = 0.f, spacing;
	const float deltaAng = .8f / ((float)NUM_ANGLE - 1);
	for(int j=0;j<NUM_ANGLE;++j) {
		px = 0.f;
		for(int i=0;i<NUM_VARIA;++i) {
			const int k = j * NUM_VARIA + i;
			if(k >= m_numPatches) {
				return;
			}
			
			tm.setIdentity();
			tm.rotateX(deltaAng * j);
			tm.setTranslation(px, py, pz);
			
			m_patches[k]->setTransformation(tm);
			
			spacing = m_patches[k]->yardRadius() * 2.f;
			px += spacing;
		}
		py += spacing * sin(deltaAng*j);
		pz -= spacing * cos(deltaAng*j);;
		
	}
}

int Vegetation::getNumInstances()
{
	int n = 0;
	for(int i=0;i<m_numPatches;++i) {
		n += m_patches[i]->getNumTms();
	}
	return n;
}

void Vegetation::clearCachedGeom()
{
	std::map<int, GeomPtrTyp >::iterator it = m_cachedGeom.begin();
	for(;it!=m_cachedGeom.end();++it) {
		delete it->second;
	}
	m_cachedGeom.clear();
	
}

ATriangleMesh * Vegetation::findGeom(const int & k)
{
	if(m_cachedGeom.find(k) != m_cachedGeom.end() ) {
		return m_cachedGeom[k];
	}
	return NULL;
}

void Vegetation::addGeom(const int & k, ATriangleMesh * v)
{ m_cachedGeom[k] = v; }

int Vegetation::numCachedGeoms() const
{ return m_cachedGeom.size(); }

void Vegetation::geomBegin(std::string & mshName, Vegetation::GeomPtrTyp & mshVal)
{
	if(numCachedGeoms() < 1) {
		mshVal = NULL;
		return;
	}
	m_geomIter = m_cachedGeom.begin();
	mshName = getGeomName(m_geomIter->first);
	mshVal = m_geomIter->second;
}
	
void Vegetation::geomNext(std::string & mshName, Vegetation::GeomPtrTyp & mshVal)
{
	m_geomIter++;
	if(m_geomIter == m_cachedGeom.end()) {
		mshVal = NULL;
		return;
	}
	mshName = getGeomName(m_geomIter->first);
	mshVal = m_geomIter->second;
}

std::string Vegetation::getGeomName(const int & k)
{
	const int gt = k>>4;
	const int gg = gar::ToGroupType(gt );
	int geomt;
	std::string geoms;
	switch (gg) {
		case gar::ggGrass:
			geomt = gar::ToGrassType(gt );
			geoms = gar::GrassTypeNames[geomt];
		break;
		default:
		;
	}
	return str(boost::format("%1%_%2%") % geoms % (k & 15));
}

void Vegetation::voxelize()
{
	for(int i=0;i<m_numPatches;++i) {
		voxelize(m_patches[i]);
	}
}

void Vegetation::voxelize(VegetationPatch * ap)
{
	sdb::VectorArray<cvx::Triangle> triangles;
	BoundingBox gridBox;
	ap->getGeom(&triangles, gridBox);
	//std::cout<<"\n Vegetation::voxelize bx"<<gridBox
	//		<<" n elm "<<triangles.size();
	//std::cout.flush();
	
	ap->setTriangleDrawCache(triangles);
	
	//ap->voxelize3(&triangles, gridBox);
}
