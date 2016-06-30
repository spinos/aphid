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
#include "tetrahedron_graph.h"
#include "tetrahedralization.h"

using namespace aphid;

namespace ttg {

TetrahedralMesher::TetrahedralMesher() :
m_X(NULL),
m_prop(NULL)
{}

TetrahedralMesher::~TetrahedralMesher() 
{ clear(); }

void TetrahedralMesher::clear()
{
	m_grid.clear();
	if(m_X) delete[] m_X;
	if(m_prop) delete[] m_prop;
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
	m_frontFaces.clear();
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
	m_prop = new int[x];
	for(int i=0; i<x; ++i)
		m_prop[i] = -1;
}

const int & TetrahedralMesher::N() const
{ return m_N; }

Vector3F * TetrahedralMesher::X()
{ return m_X; }

const Vector3F * TetrahedralMesher::X() const
{ return m_X; }

const int * TetrahedralMesher::prop() const
{ return m_prop; }

int TetrahedralMesher::build()
{
	m_grid.getNodePositions(m_X);
	m_grid.buildTetrahedrons(m_tets);
	return m_tets.size();
}

bool TetrahedralMesher::addPoint(const int & vi,
								bool & topologyChanged)
{
	Float4 coord;
	ITetrahedron * t = searchTet(m_X[vi], &coord);
	if(!t ) return false;
	
#define SNAPFACTOR .124f
	const float threshold = m_grid.gridSize() * SNAPFACTOR;
	
	topologyChanged = insertToTetrahedralMesh(m_tets, t, vi, coord, m_X, threshold,
						m_prop);
	
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
{
	const ITetrahedron * t = m_tets[vi];
	if(t->index < 0) return NULL;
	return t; 
}

const ITetrahedron * TetrahedralMesher::frontTetrahedron(const int & vi,
									int nfront,
									int nmaxfront) const
{
	const ITetrahedron * t = tetrahedron(vi);
	if(!t) return t;
	
	int c = countFrontVetices(t);
	if(c < nfront 
		|| c >= nmaxfront)
		return NULL;
		
	return t;
}

int TetrahedralMesher::countFrontVetices(const ITetrahedron * t) const
{
	int r = 0;
	if(m_prop[t->iv0] > -1) r++;
	if(m_prop[t->iv1] > -1) r++;
	if(m_prop[t->iv2] > -1) r++;
	if(m_prop[t->iv3] > -1) r++;
	return r;
}

void TetrahedralMesher::addFrontFaces(const ITetrahedron * t)
{
	ITRIANGLE trif;
	int i=0;
	for(;i<4;++i) {
		faceOfTetrahedron(&trif, t, i);
		if(m_prop[trif.p1] > 0
			&& m_prop[trif.p2] > 0
			&& m_prop[trif.p3] > 0) {
				aphid::sdb::Coord3 itri = aphid::sdb::Coord3(trif.p1, trif.p2, trif.p3).ordered();
				IFace * tri = m_frontFaces.find(itri );
				if(!tri) {
					tri = new IFace;
					tri->key = itri;
					
					m_frontFaces.insert(itri, tri);
				}
			}
	}
}

int TetrahedralMesher::buildFrontFaces()
{
	const int n = numTetrahedrons();
	int i = 0;
	for(;i<n;++i) {
		const ITetrahedron * t = frontTetrahedron(i);
		if(!t) continue;
		addFrontFaces(t);
	}
	return m_frontFaces.size();
}

sdb::Array<sdb::Coord3, IFace > * TetrahedralMesher::frontFaces()
{ return &m_frontFaces; }

/// move red to closest sample if close enought
/// move blue to closest while maintain the minimum angle
void TetrahedralMesher::moveNodesInCell(const Vector3F & c,
						const std::vector<Vector3F> & pos)
{
/// closest to red
	Vector3F redP;
	float d;
	if(GetClosest<Vector3F>(redP, d, c, pos) < 0)
		return;
		
	if(!m_grid.moveRedNodeTo(c, d, redP, m_X, m_prop) )
		redP = c;
		
	m_grid.moveBlueNodes(c, redP, pos, m_X, m_prop);
}

void TetrahedralMesher::smoothBlueNodeInCell(const aphid::Vector3F & cellCenter)
{
	m_grid.smoothBlueNodeIn(cellCenter, m_X);
}

}