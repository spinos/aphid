/*
 *  Hilbert2D.cpp
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Hilbert2D.h"
#include <CartesianGrid.h>
#include <iostream>
#include "hilbertCurve.h"

using namespace aphid;
namespace ttg {

Hilbert2D::Hilbert2D()
{}

Hilbert2D::~Hilbert2D() 
{
	delete[] m_X;
	delete[] m_ind;
}

bool Hilbert2D::init() 
{
	generateSamples();
    return true;
}

void Hilbert2D::generateSamples()
{
	std::cout<<" generate samples by 32 X 32 grid\n";
	BoundingBox bbx(0.f, 0.f, 0.f,
					32.f, 32.f, 32.f);
	const float x0 = 16.f;
	const float y0 = 16.f;
	const float xRed = 16.f;
	const float yRed = 0.f;
	const float xBlue = 0.f;
	const float yBlue = 16.f;
	
	CartesianGrid bgg;
	bgg.setBounding(bbx);
	
	int level = 5;
    const int dim = 1<<level;
    int i, j;

    const float h = bgg.cellSizeAtLevel(level);
    const float hh = h * .5f;
    
    const Vector3F ori = bgg.origin() + Vector3F(hh, hh, hh) * .999f;
    Vector3F sample, closestP;
    for(j=0; j < dim; j++) {
		for(i=0; i < dim; i++) {
			sample = ori + Vector3F(h* (float)i, h* (float)j, (float)h);
			if(RandomF01() < .97f )
				bgg.addCell(sample, level, 0);
		}
	}
	m_N = bgg.numCells();
	std::cout<<" n background cell "<<m_N<<std::endl;
	
	m_X = new Vector3F[m_N];
	m_ind = new QuickSortPair<int, int>[m_N];
	
	i = 0;
	
	sdb::CellHash * cel = bgg.cells();
	cel->begin();
	while(!cel->end()) {
		sample = bgg.cellCenter(cel->key());
		sample.x += RandomFn11() * 0.33f * h;
		sample.y += RandomFn11() * 0.33f * h;
		sample.z = 0.f;
		
		m_ind[i].key = hilbert2DCoord(sample.x, sample.y, 
									x0, y0,
									xRed, yRed,
									xBlue, yBlue,
									level);
		
		m_ind[i].value = i;
		m_X[i++].set(sample.x, sample.y, 0.f);
	    cel->next();
	}
	
	QuickSort1::Sort<int, int>(m_ind, 0, m_N-1);
}

bool Hilbert2D::progressForward()
{ 
	return true; 
}

bool Hilbert2D::progressBackward()
{ 
	return true; 
}

const char * Hilbert2D::titleStr() const
{ return "Hilbert 2D Test"; }

void Hilbert2D::draw(GeoDrawer * dr) 
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