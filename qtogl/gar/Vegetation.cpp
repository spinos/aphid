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
#include <iomanip>
#include <cmath>

using namespace aphid;

Vegetation::Vegetation() :
m_numPatches(1)
{
	for(int i=0;i<TOTAL_NUM_PAC;++i) {
		m_patches[i] = new VegetationPatch;
	}
}

Vegetation::~Vegetation()
{
	for(int i=0;i<TOTAL_NUM_PAC;++i) {
		delete m_patches[i];
	}
	clearCachedGeom();
}

void Vegetation::setSynthByAngleAlign()
{
	Variform::setPattern(Variform::pnAngleAlign);
	const float deltaAng = deltaAnglePerGroup();
	for(int j=0;j<NumAngleGroups;++j) {
		const float angj = deltaAng * j;
		for(int i=0;i<NumEventsPerGroup;++i) {
			m_patches[j * NumEventsPerGroup + i]->setTilt(angj);
		}
	}
}

void Vegetation::setSynthByRandom()
{
	Variform::setPattern(Variform::pnRandom);
	for(int i=0;i<TOTAL_NUM_PAC;++i) {
		m_patches[i]->setTilt(0.f);
	}
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
{ return TOTAL_NUM_PAC; }

void Vegetation::rearrange()
{
	Matrix44F tm;
	float px, pz = 0.f, py = 0.f, spacing;
	const float deltaAng = deltaAnglePerGroup();
	for(int j=0;j<NumAngleGroups;++j) {
		px = 0.f;
		for(int i=0;i<NumEventsPerGroup;++i) {
			const int k = j * NumEventsPerGroup + i;
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

int Vegetation::getGeomInd(aphid::ATriangleMesh * x)
{
	int i = 0;
	std::map<int, GeomPtrTyp >::iterator it = m_cachedGeom.begin();
	for(;it!=m_cachedGeom.end();++it) {
		if(x == it->second) {
			return i;
		}
		i++;
	}
	return 0;
}

int Vegetation::numCachedGeoms() const
{ return m_cachedGeom.size(); }

void Vegetation::geomBegin(std::string & mshName, Vegetation::GeomPtrTyp & mshVal)
{
	if(numCachedGeoms() < 1) {
		mshVal = NULL;
		return;
	}
	m_geomIter = m_cachedGeom.begin();
	m_curGeomId = 0;
	mshName = getGeomName(m_geomIter->first);
	mshVal = m_geomIter->second;
}
	
void Vegetation::geomNext(std::string & mshName, Vegetation::GeomPtrTyp & mshVal)
{
	m_geomIter++;
	m_curGeomId++;
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
	return str(boost::format("inst_%1%_%2%_%3%") % boost::io::group(std::setw(3), std::setfill('0'), m_curGeomId) % geoms % (k & 15));
}

void Vegetation::voxelize()
{
	m_bbox.reset();
	for(int i=0;i<m_numPatches;++i) {
		voxelize(m_patches[i]);
		m_bbox.expandBy(m_patches[i]->geomBox() );
	}
	std::cout<<"\n vegetation bbox "<<m_bbox;
	std::cout.flush();
	
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
	
	ap->voxelize3(&triangles, gridBox);
}

const aphid::BoundingBox & Vegetation::bbox() const
{ return m_bbox; }
