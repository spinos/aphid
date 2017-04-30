/*
 *  Delaunay2D.cpp
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Delaunay2D.h"
#include <CartesianGrid.h>
#include <iostream>
#include "hilbertCurve.h"

using namespace aphid;
namespace ttg {

Delaunay2D::Delaunay2D() :
m_endTri(100) 
{}

Delaunay2D::~Delaunay2D() 
{
	delete[] m_X;
	delete[] m_ind;
	delete[] m_triangles;
}

bool Delaunay2D::init() 
{
	generateSamples();
	triangulate();
    return true;
}

void Delaunay2D::generateSamples()
{
	std::cout<<" generate samples by 16 X 16 grid\n";
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
	
	int level = 4;
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
	
/// first three for super triangle
	m_X = new Vector3F[m_N+3];
	m_ind = new QuickSortPair<int, int>[m_N+3];
	m_triangles = new ITRIANGLE[m_N * 3 + 9];
	float dx = bbx.distance(0) + bbx.distance(1);
	m_X[0].set(bbx.getMin(0) - dx * 3.f, bbx.getMin(1) - dx, 0.f);
	m_X[1].set(bbx.getMax(0) + dx, bbx.getMin(1) - dx, 0.f);
	m_X[2].set(bbx.getMax(0) + dx, bbx.getMax(1) + dx * 3.f, 0.f);
	m_ind[0].value = 0;
	m_ind[1].value = 1;
	m_ind[2].value = 2;
	i = 3;
	
	sdb::CellHash * cel = bgg.cells();
	cel->begin();
	while(!cel->end()) {
		sample = bgg.cellCenter(cel->key());
		sample.x += RandomFn11() * 0.43f * hh;
		sample.y += RandomFn11() * 0.43f * hh;
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
	QuickSort1::Sort<int, int>(m_ind, 3, m_N+3-1);
}

bool Delaunay2D::triangulate()
{
	ITRIANGLE superTri;
	superTri.p1 = 0;
	superTri.p2 = 1;
	superTri.p3 = 2;
	m_numTri = 0;
	m_triangles[m_numTri++] = superTri;
	
	int i = 3;
	for(;i<m_endTri;++i) {
/// Lawson's find the triangle that contains X[i]
		std::cout<<"\n insert X["<<i<<"]\n";
		int ii = m_ind[i].value;
		int j = searchTri(m_X[ii]);
		std::cout<<"\n insie tri["<<j<<"]\n";
		
		splitTriangle(&m_triangles[j], &m_triangles[m_numTri], &m_triangles[m_numTri+1],
				ii);
/// add two more triangles				
		m_numTri+=2;
				
		std::deque<Quadrilateral> qls;
		Quadrilateral q1;
		q1.ta = &m_triangles[j];
		q1.tb = m_triangles[j].nei[0];
		q1.apex = ii;
		findAntiApex(q1);
		qls.push_back(q1);
		flipEdges(qls, m_X);
		
		Quadrilateral q2;
		q2.ta = &m_triangles[m_numTri-2];
		q2.tb = m_triangles[m_numTri-2].nei[0];
		q2.apex = ii;
		findAntiApex(q2);
		qls.push_back(q2);
		flipEdges(qls, m_X);
		
		Quadrilateral q3;
		q3.ta = &m_triangles[m_numTri-1];
		q3.tb = m_triangles[m_numTri-1].nei[0];
		q3.apex = ii;
		findAntiApex(q3);
		qls.push_back(q3);
		flipEdges(qls, m_X);
		
	}
	std::cout<<" end triangulate X["<<i-1<<"]"<<std::endl;
	return true;
}

bool Delaunay2D::progressForward()
{ 
	m_endTri++;
	if(m_endTri>m_N+3) m_endTri=m_N+3;
	return triangulate(); 
}

bool Delaunay2D::progressBackward()
{ 
	m_endTri--;
	if(m_endTri<5) m_endTri=5;
	return triangulate(); 
}

const char * Delaunay2D::titleStr() const
{ return "Delaunay 2D Test"; }

void Delaunay2D::draw(GeoDrawer * dr) 
{
	dr->setColor(0.f, 0.f, 0.f);
	int i = 3;
	for(;i<m_N+3;++i) {
		dr->cube(m_X[i], .25f);
	}
	dr->setColor(0.2f, 0.2f, 0.4f);
	glBegin(GL_LINES);
	Vector3F a, b, c;
	i = 0;
	for(;i<m_numTri;++i) {
		const ITRIANGLE t = m_triangles[i];
		a = m_X[t.p1];
		b = m_X[t.p2];
		c = m_X[t.p3];
		glVertex3fv((const GLfloat *)&a);
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&b);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&c);
		glVertex3fv((const GLfloat *)&a);
	}
	glEnd();
#if 0
	dr->setColor(.5f, .5f, .5f);
	TriCircle cir;
	Matrix44F rot;
	i = 0;
	for(;i<m_numTri;++i) {
		const ITRIANGLE t = m_triangles[i];
		a = m_X[t.p1];
		b = m_X[t.p2];
		c = m_X[t.p3];
		
		circumCircle(cir, a, b, c);
		rot.setTranslation(cir.pc);
		dr->circleAt(rot, cir.r);
	}
#endif
}

int Delaunay2D::searchTri(const aphid::Vector3F & p) const
{
	Vector3F a, b, c;
	int i=m_numTri-1;
	for(; i>=0; --i) {
		const ITRIANGLE t = m_triangles[i];
		a = m_X[t.p1];
		b = m_X[t.p2];
		c = m_X[t.p3];
		
		if(insideTri(p, a, b, c) ) return i;
	}
	return 0;
}

}