/*
 *  PlantPiece.cpp
 *  garden
 *
 *  Created by jian zhang on 4/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlantPiece.h"
#include <geom/ATriangleMesh.h>
#include <geom/ConvexShape.h>
#include <sdb/VectorArray.h>

using namespace aphid;

PlantPiece::PlantPiece(PlantPiece * parent) :
m_parentPiece(parent),
m_geom(NULL),
m_exclR(1.f),
m_geomid(0)
{
	if(parent) {
		parent->addBranch(this);
	}
}

PlantPiece::~PlantPiece()
{
	ChildListTyp::iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		delete *it;
	}
	m_childPieces.clear();
}

void PlantPiece::addBranch(PlantPiece * c)
{ m_childPieces.push_back(c); }

void PlantPiece::setTransformMatrix(const Matrix44F &tm)
{ m_tm = tm; }

const Matrix44F & PlantPiece::transformMatrix() const
{ return m_tm; }

int PlantPiece::numBranches() const
{ return m_childPieces.size(); }

const PlantPiece * PlantPiece::branch(const int & i) const
{ return m_childPieces[i]; }

void PlantPiece::setGeometry(ATriangleMesh * geom, const int & geomId)
{ 
	m_geom = geom; 
	m_geomid = geomId;
}

const ATriangleMesh * PlantPiece::geometry() const
{ return m_geom; }

void PlantPiece::setExclR(const float & x)
{ m_exclR = x; }
	
const float & PlantPiece::exclR() const
{ return m_exclR; }

void PlantPiece::setExclRByChild()
{
	m_exclR = 0.f;
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		const float & r = (*it)->exclR();
		if(m_exclR < r) {
			m_exclR = r;
		}
	}
	
}

void PlantPiece::countNumTms(int & count) const
{
	if(m_geom) {
		count++;
	}
	
	if(numBranches() < 1) {
		return;
	}
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->countNumTms(count);
	}
}

void PlantPiece::extractTms(aphid::Matrix44F * dst,
			int & count) const
{
	if(m_geom) {
		Matrix44F & mat = dst[count];
		mat.setIdentity();
		worldTransformMatrix(mat);
		count++;
	}
	
	if(numBranches() < 1) {
		return;
	}
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->extractTms(dst, count);
	}
}

void PlantPiece::extractGeomIds(int * dst,
			int & count) const
{
	if(m_geom) {
		dst[count] = m_geomid;
		count++;
	}
	
	if(numBranches() < 1) {
		return;
	}
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->extractGeomIds(dst, count);
	}
}

void PlantPiece::worldTransformMatrix(Matrix44F & dst) const
{
	dst.multiply(m_tm);
	if(m_parentPiece) {
		m_parentPiece->worldTransformMatrix(dst);
	}
}

void PlantPiece::getGeom(GeomElmArrTyp * dst,
					BoundingBox & box,
					const aphid::Matrix44F & relTm)
{
	Matrix44F wtm;
	worldTransformMatrix(wtm);
	wtm.multiply(relTm);
	
	getGeomElm(dst, box, wtm);
	
	if(numBranches() < 1) {
		return;
	}
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->getGeom(dst, box, relTm);
	}
	
}

void PlantPiece::getGeomElm(GeomElmArrTyp * dst,
					BoundingBox & box,
					const Matrix44F & relTm)
{
	if(!m_geom) {
		return;
	}
	GeomElmTyp acomp;
	const int n = m_geom->numComponents();
	for(int j=0; j<n; ++j) {
		
		m_geom->dumpComponent<GeomElmTyp>(acomp, j, relTm);
		dst->insert(acomp);
			
		const BoundingBox cbx = acomp.calculateBBox();
		box.expandBy(cbx);
	}
	
}
