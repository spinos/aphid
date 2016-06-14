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

Bcc3dTest::Bcc3dTest()
{}

Bcc3dTest::~Bcc3dTest() 
{}

bool Bcc3dTest::init() 
{ createGrid(); }

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
	m_grid.buildTetrahedrons();
	std::cout<<"\n n node "<<m_grid.numNodes();
	std::cout.flush();
}

}
