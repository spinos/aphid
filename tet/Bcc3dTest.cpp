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
#include "tetrahedralization.h"

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
	
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	
	std::vector<ITetrahedron *>::const_iterator it = m_tets.begin();
	for(;it!= m_tets.end();++it) {
		const ITetrahedron * t = *it;
		
		a = m_X[t->iv0];
		b = m_X[t->iv1];
		c = m_X[t->iv2];
		d = m_X[t->iv3];
		
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&d);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&b);
		
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&d);
		glVertex3fv((const GLfloat *)&c);
	}
	
	glEnd();
}

void Bcc3dTest::createGrid()
{
	const float h = 8.f;
	m_grid.setGridSize(h);
    int dimx = 2;
	int dimy = 1;
	int dimz = 1;
	std::cout<<" generate samples by "<<dimx<<" X "<<dimy<<" X "<<dimz<<" grid\n";
	int i, j, k;

    const float hh = 4.f;
    
    const Vector3F ori(1.f, 1.f, 1.f);
    Vector3F sample;
	for(k=0; k < dimz; k++) {
		for(j=0; j < dimy; j++) {
			for(i=0; i < dimx; i++) {
				sample = ori + Vector3F(h* (float)i, h* (float)j, h*(float)k);
/// center of cell				
				BccNode * node15 = new BccNode;
				node15->key = 15;
				m_grid.insert((const float *)&sample, node15 );
			}
		}
	}
	m_grid.calculateBBox();
	std::cout<<"\n n cell "<<m_grid.size();
	m_grid.buildNodes();
	
#define ADDON 13
	m_N = m_grid.numNodes() + ADDON;
	m_X = new Vector3F[m_N];
	std::cout<<"\n n node "<<m_N;
	m_grid.getNodePositions(m_X);
	m_grid.buildTetrahedrons(m_tets);
	
/// on edge
	i = m_N - ADDON;
	
	m_X[i].set(2.f, 2.f, 2.f);
	m_X[i+1].set(3.f, 3.f, 5.f);
	m_X[i+2].set(2.f, 6.f, 6.f);
	m_X[i+3].set(2.f, 6.f, 2.f);
	
	m_X[i+4].set(6.f, 6.f, 1.f);
	m_X[i+5].set(6.f, 6.f, 3.3f);
	m_X[i+6].set(8.91f, 6.2f, 3.3f);
	m_X[i+7].set(10.f, 6.f, 2.f);
	
	m_X[i+8].set(6.f, 2.f, 6.f);
	m_X[i+9].set(6.f, 2.f, 2.f);
	m_X[i+10].set(10.f, 2.f, 6.f);
	m_X[i+11].set(14.f, 1.f, 2.f);
	
	m_X[i+12].set(14.71f, 4.74f, 5.93f);
	
	i = m_N - ADDON+4;
	for(;i<m_N - ADDON+5;++i) {
		addPoint(i);
		checkTetrahedronConnections(m_tets);
	}
	
	std::cout<<"\n n tet "<<m_tets.size();
	std::cout.flush();
	
}

bool Bcc3dTest::addPoint(const int & vi)
{
	Float4 coord;
	ITetrahedron * t = searchTet(m_X[vi], &coord);
	if(!t ) return false;
	splitTetrahedron(m_tets, t, vi, coord);
	
	return true;
}

ITetrahedron * Bcc3dTest::searchTet(const aphid::Vector3F & p, Float4 * coord)
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

}
