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
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<m_N;++i) {
		dr->cube(m_X[i], .125f);
		dr->drawNumber(i, m_X[i], .5f);
	}
	
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	
	std::vector<ITetrahedron *>::const_iterator it = m_tets.begin();
	for(;it!= m_tets.end();++it) {
		const ITetrahedron * t = *it;
		if(t->index < 0) continue;
		
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
	
#define ADDON 16
	m_N = m_grid.numNodes() + ADDON;
	m_X = new Vector3F[m_N];
	std::cout<<"\n n node "<<m_N;
	m_grid.getNodePositions(m_X);
	m_grid.buildTetrahedrons(m_tets);
	
/// on edge
	i = m_N - ADDON;
	
	m_X[i].set(3.2f, 3.2f, 3.2f); // edge
	m_X[i+1].set(3.f, 3.f, 5.f); // edge
	m_X[i+2].set(2.f, 6.f, 6.f); // edge
	m_X[i+3].set(3.f, 5.f, 3.f); // edge 4
	m_X[i+4].set(5.2f, 5.2f, .9f); // face
	m_X[i+5].set(6.f, 6.f, 3.3f); // face 6
	
	m_X[i+6].set(5.11f, 5.94f, 6.2f); // inside
	m_X[i+7].set(7.11f, 6.f, 2.f); // inside 8
	m_X[i+8].set(10.f, 6.17f, 4.93f);
	
	m_X[i+9].set(3.53f, 5.67f, 5.63f);
	m_X[i+10].set(8.01f, 6.f, 6.f);
	m_X[i+11].set(9.f, 6.391f, 2.35f);
	
	m_X[i+12].set(11.1f, 6.191f, 2.35f);
	m_X[i+13].set(13.1f, 6.091f, 3.435f);
	m_X[i+14].set(12.1f, 5.791f, 5.335f);
	m_X[i+15].set(11.1f, 4.8291f, 6.035f);
	
#define ENDN 16
	i = m_N - ADDON + 0;
	for(;i<m_N - ADDON+ENDN;++i) {
		addPoint(i);
		if(!checkTetrahedronConnections(m_tets) ) {
			std::cout<<"\n [WARNING] break at v"<<i;
			break;
		}
	}
	
	std::cout<<"\n n tet "<<m_tets.size();
	std::cout.flush();
	
}

bool Bcc3dTest::addPoint(const int & vi)
{
	Float4 coord;
	ITetrahedron * t = searchTet(m_X[vi], &coord);
	if(!t ) return false;
	splitTetrahedron(m_tets, t, vi, coord, m_X);
	
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
