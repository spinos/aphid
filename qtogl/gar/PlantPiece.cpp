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

using namespace aphid;

PlantPiece::PlantPiece(PlantPiece * parent) :
m_parentPiece(parent),
m_geom(NULL),
m_exclR(1.f)
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

void PlantPiece::setGeometry(aphid::ATriangleMesh * geom)
{ m_geom = geom; }

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
	if(numBranches() < 1) {
		return;
	}
	count++;
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->countNumTms(count);
	}
}

void PlantPiece::extractTms(aphid::Matrix44F * dst,
			int & count) const
{
	if(numBranches() < 1) {
		return;
	}
	dst[count] = m_tm;
	count++;
	
	ChildListTyp::const_iterator it = m_childPieces.begin();
	for(;it!=m_childPieces.end();++it) {
		(*it)->extractTms(dst, count);
	}
}
