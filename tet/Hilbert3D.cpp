/*
 *  Hilbert3D.cpp
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Hilbert3D.h"
#include <CartesianGrid.h>
#include <iostream>
#include "hilbertCurve.h"

using namespace aphid;
namespace ttg {

Hilbert3D::Hilbert3D() :
m_level(1),
m_X(NULL),
m_ind(NULL)
{}

Hilbert3D::~Hilbert3D() 
{
	delete[] m_X;
	delete[] m_ind;
}

bool Hilbert3D::init() 
{
	generateSamples(m_level);
    return true;
}

void Hilbert3D::generateSamples(int level)
{
	BoundingBox bbx(0.f, 0.f, 0.f,
					32.f, 32.f, 32.f);
	const float x0 = 16.f;
	const float y0 = 16.f;
	const float z0 = 16.f;
	const float xRed = 16.f;
	const float yRed = 0.f;
	const float zRed = 0.f;
	const float xGreen = 0.f;
	const float yGreen = 16.f;
	const float zGreen = 0.f;
	const float xBlue = 0.f;
	const float yBlue = 0.f;
	const float zBlue = 16.f;
	
	CartesianGrid bgg;
	bgg.setBounding(bbx);
	
    const int dim = 1<<level;
	
	std::cout<<" generate samples by "<<dim<<" X "<<dim<<" X "<<dim<<" grid\n";
	
    int i, j, k;

    const float h = bgg.cellSizeAtLevel(level);
    const float hh = h * .5f;
    
    const Vector3F ori = bgg.origin() + Vector3F(hh, hh, hh) * .999f;
    Vector3F sample, closestP;
    for(k=0; k < dim; k++) {
		for(j=0; j < dim; j++) {
			for(i=0; i < dim; i++) {
				sample = ori + Vector3F(h* (float)i, h* (float)j, h * (float)k);
				bgg.addCell(sample, level, 0);
			}
		}
	}
	m_N = bgg.numCells();
	std::cout<<" n background cell "<<m_N<<std::endl;
	
	if(m_X) delete[] m_X;
	if(m_ind) delete[] m_ind;
	m_X = new Vector3F[m_N];
	m_ind = new QuickSortPair<int, int>[m_N];
	
	i = 0;
	
	sdb::CellHash * cel = bgg.cells();
	cel->begin();
	while(!cel->end()) {
		sample = bgg.cellCenter(cel->key());
		sample.x += RandomFn11() * 0.3f * hh;
		sample.y += RandomFn11() * 0.3f * hh;
		sample.z += RandomFn11() * 0.3f * hh;
		
		m_ind[i].key = hilbert3DCoord(sample.x, sample.y, sample.z,
									x0, y0, z0,
									xRed, yRed, zRed,
									xGreen, yGreen, zGreen,
									xBlue, yBlue, zBlue,
									level);
		
		m_ind[i].value = i;
		m_X[i++].set(sample.x, sample.y, sample.z);
	    cel->next();
	}
	
	QuickSort1::Sort<int, int>(m_ind, 0, m_N-1);
}

bool Hilbert3D::progressForward()
{ 
	m_level++;
	if(m_level > 5) m_level = 5;
	generateSamples(m_level);
	return true; 
}

bool Hilbert3D::progressBackward()
{ 
	m_level--;
	if(m_level < 1) m_level = 1;
	generateSamples(m_level);
	return true; 
}

const char * Hilbert3D::titleStr() const
{ return "Hilbert 3D Test"; }

void Hilbert3D::draw(GeoDrawer * dr) 
{
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<m_N;++i) {
		dr->cube(m_X[i], .25f);
	}
	dr->setColor(0.2f, 0.2f, 0.4f);
	glBegin(GL_LINES);
	Vector3F a, b, c;
	i = 1;
	for(;i<m_N;++i) {
		a = m_X[m_ind[i-1].value ];
		b = m_X[m_ind[i].value ];
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&b);
	}
	glEnd();
}

}