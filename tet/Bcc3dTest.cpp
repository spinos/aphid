/*
 *  Bcc3dTest.cpp
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Bcc3dTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

Bcc3dTest::Bcc3dTest() :
m_X(NULL)
{}

Bcc3dTest::~Bcc3dTest() 
{
	if(m_X) delete[] m_X;
	std::vector<ITetrahedron *>::iterator it = m_tets.begin();
	for(;it!=m_tets.end();++it) {
		delete *it;
	}
	m_tets.clear();
}

bool Bcc3dTest::init() 
{ 
    createGrid();
    return true;
}

bool Bcc3dTest::progressForward()
{ 
	return true; 
}

bool Bcc3dTest::progressBackward()
{ 
	return true; 
}

const char * Bcc3dTest::titleStr() const
{ return "BCC 3D Test"; }

void Bcc3dTest::draw(GeoDrawer * dr) 
{
	dr->setColor(0.3f, 0.3f, 0.39f);
	m_grid.begin();
	while(!m_grid.end() ) {
		dr->boundingBox(m_grid.coordToGridBBox(m_grid.key() ) );
		m_grid.next();
	}
	
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<m_N;++i) {
		dr->cube(m_X[i], .125f);
	}
}

void Bcc3dTest::createGrid()
{
	m_grid.setGridSize(4.f);
	int dim = 1<<2;
	std::cout<<" generate samples by "<<dim<<" X "<<dim<<" X "<<dim<<" grid\n";
	int i, j, k;

    const float h = 4.f;
    const float hh = 2.f;
    
    const Vector3F ori(-1.5f, -1.5f, -1.5f);
    Vector3F sample;
	for(k=0; k < dim; k++) {
		for(j=0; j < dim; j++) {
			for(i=0; i < dim; i++) {
				sample = ori + Vector3F(h* (float)i, h* (float)j, h*(float)k);
/// center of cell				
				BccNode * nouse = new BccNode;
				nouse->key = 15;
				m_grid.insert((const float *)&sample, nouse );
			}
		}
	}
	m_grid.calculateBBox();
	std::cout<<"\n n cell "<<m_grid.size();
	m_grid.buildNodes();
	m_N = m_grid.numNodes();
	m_X = new Vector3F[m_N];
	std::cout<<"\n n node "<<m_N;
	m_grid.getNodePositions(m_X);
	m_grid.buildTetrahedrons(m_tets);
	std::cout<<"\n n tet "<<m_tets.size();
	std::cout.flush();
	
}

}
