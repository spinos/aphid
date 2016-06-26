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
	const int Nv = m_mesher.N();
	const int Nt = m_mesher.numTetrahedrons();
	const Vector3F * X = m_mesher.X();
	
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<Nv;++i) {
		dr->cube(X[i], .125f);
		dr->drawNumber(i, X[i], .5f);
	}
	
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.49f);
	
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	
	for(i=0; i<Nt; ++i) {
		const ITetrahedron * t = m_mesher.tetrahedron(i);
		if(t->index < 0) continue;
		
		a = X[t->iv0];
		b = X[t->iv1];
		c = X[t->iv2];
		d = X[t->iv3];
		
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
	m_mesher.setH(h);
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
				m_mesher.addCell(sample);
			}
		}
	}
	
	int nbcc = m_mesher.finishGrid();
	
#define ADDON 16
	m_mesher.setN(nbcc + ADDON );
	
	int N = m_mesher.N();
	std::cout<<"\n n node "<<N;
	
	std::cout<<"\n n grid tetra "<<m_mesher.build();
	
/// on edge
	i = N - ADDON;
	
	Vector3F * X = m_mesher.X();
	
	X[i].set(3.2f, 3.2f, 3.2f); // edge
	X[i+1].set(3.f, 3.f, 5.f); // edge
	X[i+2].set(2.f, 6.f, 6.f); // edge
	X[i+3].set(3.f, 5.f, 3.f); // edge 4
	X[i+4].set(5.2f, 5.2f, .9f); // face
	X[i+5].set(6.f, 6.f, 3.3f); // face 6
	
	X[i+6].set(5.11f, 5.94f, 6.2f); // inside
	X[i+7].set(7.11f, 6.f, 2.f); // inside 8
	X[i+8].set(10.f, 6.17f, 4.93f);
	
	X[i+9].set(3.53f, 5.67f, 5.63f);
	X[i+10].set(8.01f, 6.f, 6.f);
	X[i+11].set(9.f, 6.391f, 2.35f);
	
	X[i+12].set(11.1f, 6.191f, 2.35f);
	X[i+13].set(13.1f, 6.091f, 3.435f);
	X[i+14].set(12.1f, 5.791f, 5.335f);
	X[i+15].set(11.1f, 4.8291f, 6.035f);
	
	bool topoChanged;
#define ENDN 16
	i = N - ADDON + 0;
	for(;i<N - ADDON+ENDN;++i) {
		if(!m_mesher.addPoint(i, topoChanged) ) {
			std::cout<<"\n [WARNING] add pnt break at v"<<i;
			break;
		}
		if(topoChanged) {
			if(!m_mesher.checkConnectivity() ) {
			std::cout<<"\n [WARNING] check conn break at v"<<i;
			break;
			}
			std::cout<<"\n passed topology check";
		}
	}
	
	std::cout<<"\n n tet "<<m_mesher.numTetrahedrons();
	std::cout.flush();
	
}

}
