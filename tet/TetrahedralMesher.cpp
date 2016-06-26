/*
 *  TetrahedralMesher.cpp
 *  foo
 *
 *  Created by jian zhang on 6/26/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetrahedralMesher.h"
#include <iostream>
#include "tetrahedralization.h"

using namespace aphid;

namespace ttg {

TetrahedralMesher::TetrahedralMesher() :
m_X(NULL) 
{}

TetrahedralMesher::~TetrahedralMesher() 
{ clear(); }

void TetrahedralMesher::clear()
{
	m_grid.clear();
	if(m_X) delete[] m_X;
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
}

void TetrahedralMesher::setH(const float & x)
{
	clear();
	m_grid.setGridSize(x);
}

void TetrahedralMesher::addCell(const Vector3F & p)
{
	if(!m_grid.findCell((const float *)&p ) ) {
		BccNode * node15 = new BccNode;
		node15->key = 15;
		m_grid.insert((const float *)&p, node15 );
	}
}

int TetrahedralMesher::finishGrid()
{
	m_grid.calculateBBox();
	std::cout<<"\n mesher grid n cell "<<m_grid.size()
			<<"\n bbx "<<m_grid.boundingBox();
	m_grid.buildNodes();
	return m_grid.numNodes();
}

int TetrahedralMesher::numNodes()
{ return m_grid.numNodes(); }

void TetrahedralMesher::setN(const int & x)
{
	m_N = x;
	m_X = new Vector3F[x];
}

const int & TetrahedralMesher::N() const
{ return m_N; }

Vector3F * TetrahedralMesher::X()
{ return m_X; }

const Vector3F * TetrahedralMesher::X() const
{ return m_X; }

int TetrahedralMesher::build()
{
	m_grid.getNodePositions(m_X);
	m_grid.buildTetrahedrons(m_tets);
	return m_tets.size();
}

bool TetrahedralMesher::addPoint(const int & vi)
{
	Float4 coord;
	ITetrahedron * t = searchTet(m_X[vi], &coord);
	if(!t ) return false;
	splitTetrahedron(m_tets, t, vi, coord, m_X);
	
	return true;
}

ITetrahedron * TetrahedralMesher::searchTet(const Vector3F & p, Float4 * coord)
{
	Vector3F v[4];
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!= m_tets.end();++it) {
		ITetrahedron * t = *it;
		if(t->index < 0) continue;
		v[0] = m_X[t->iv0];
		v[1] = m_X[t->iv1];
		v[2] = m_X[t->iv2];
		v[3] = m_X[t->iv3];
		if(pointInsideTetrahedronTest1(p, v, coord) ) return t;
	}
	return NULL;
}

bool TetrahedralMesher::checkConnectivity()
{ return checkTetrahedronConnections(m_tets); }

int TetrahedralMesher::numTetrahedrons()
{ return m_tets.size(); }

const ITetrahedron * TetrahedralMesher::tetrahedron(const int & vi) const
{ return m_tets[vi]; }

}