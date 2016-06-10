/*
 *  Delaunay3D.cpp
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Delaunay3D.h"
#include <CartesianGrid.h>
#include <iostream>
#include "hilbertCurve.h"

using namespace aphid;
namespace ttg {

Delaunay3D::Delaunay3D() :
m_endTet(5) 
{}

Delaunay3D::~Delaunay3D() 
{
	delete[] m_X;
	delete[] m_ind;
	delete[] m_tets;
}

bool Delaunay3D::init() 
{
	generateSamples();
	tetrahedralize();
}

void Delaunay3D::generateSamples()
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
	
	int level = 3;
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
				sample = ori + Vector3F(h* (float)i, h* (float)j, h*(float)k);
				if(RandomF01() < .93f )
					bgg.addCell(sample, level, 0);
			}
		}
	}
	m_N = bgg.numCells();
	std::cout<<" n background cell "<<m_N<<std::endl;
	
/// first four for super tetrahedron
	m_X = new Vector3F[m_N+4];
	m_ind = new QuickSortPair<int, int>[m_N+4];
	m_tets = new ITetrahedron[m_N * 4 + 16];
	const float dx = bbx.distance(0) + bbx.distance(1) + bbx.distance(2);
	const float dx3 = dx * 3.f;
	m_X[0].set(bbx.getMax(0) + dx, bbx.getMax(1) + dx3, bbx.getMin(2) - dx);
	m_X[1].set(bbx.getMax(0) + dx, bbx.getMin(1) - dx, bbx.getMin(2) - dx);
	m_X[2].set(bbx.getMax(0) + dx, bbx.getMin(1) - dx, bbx.getMax(2) + dx3);
	m_X[3].set(bbx.getMin(0) - dx3, bbx.getMin(1) - dx, bbx.getMin(2) - dx);
	m_ind[0].value = 0;
	m_ind[1].value = 1;
	m_ind[2].value = 2;
	m_ind[3].value = 3;
	i = 4;
	
	sdb::CellHash * cel = bgg.cells();
	cel->begin();
	while(!cel->end()) {
		sample = bgg.cellCenter(cel->key());
		sample.x += RandomFn11() * 0.43f * hh;
		sample.y += RandomFn11() * 0.43f * hh;
		sample.z += RandomFn11() * 0.43f * hh;
		
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
	QuickSort1::Sort<int, int>(m_ind, 4, m_N+4-1);
}

bool Delaunay3D::tetrahedralize()
{
	ITetrahedron superTet;
	setTetrahedronVertices(superTet, 0, 1, 2, 3);
	resetTetrahedronNeighbors(superTet);
	m_numTet = 0;
	m_tets[m_numTet++] = superTet;
	
	int i = 4;
	for(;i<m_endTet;++i) {
/// Lawson's find the tetrahedron that contains X[i]
		std::cout<<"\n insert X["<<i<<"]\n";
		const int ii = m_ind[i].value;
		std::cout<<"\n ind "<<ii;
		int j = searchTet(m_X[ii]);
		ITetrahedron t = m_tets[j];

		splitTetrahedron(&m_tets[j], &m_tets[m_numTet], &m_tets[m_numTet+1], &m_tets[m_numTet+2], ii);
/// add three tetrahedrons
		m_numTet+=3;		
/*				
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
		*/
	}
	std::cout<<" end triangulate X["<<i-1<<"]"<<std::endl;
	return true;
}

bool Delaunay3D::progressForward()
{ 
	m_endTet++;
	if(m_endTet>m_N+4) m_endTet=m_N+4;
	return tetrahedralize(); 
}

bool Delaunay3D::progressBackward()
{ 
	m_endTet--;
	if(m_endTet<5) m_endTet=5;
	return tetrahedralize(); 
}

const char * Delaunay3D::titleStr() const
{ return "Delaunay 3D Test"; }

void Delaunay3D::draw(GeoDrawer * dr) 
{
	dr->m_markerProfile.apply();
	dr->setColor(0.f, 0.f, 0.f);
	int i = 3;
	for(;i<m_N+4;++i) {
		dr->cube(m_X[i], .25f);
	}
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.4f);
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	i = 0;
	for(;i<m_numTet;++i) {
		const ITetrahedron t = m_tets[i];
		a = m_X[t.iv0];
		b = m_X[t.iv1];
		c = m_X[t.iv2];
		d = m_X[t.iv3];
		
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
	
#if 1
	dr->setColor(.5f, .5f, .5f);
	TetSphere cir;
	i = 0;
	for(;i<m_numTet;++i) {
		const ITetrahedron t = m_tets[i];
		a = m_X[t.iv0];
		b = m_X[t.iv1];
		c = m_X[t.iv2];
		d = m_X[t.iv3];
		
		circumSphere(cir, a, b, c, d);
		dr->alignedCircle(cir.pc, cir.r);
	}
#endif
}

int Delaunay3D::searchTet(const aphid::Vector3F & p) const
{
	Vector3F v[4];
	int i=m_numTet-1;
	for(; i>=0; --i) {
		const ITetrahedron t = m_tets[i];
		v[0] = m_X[t.iv0];
		v[1] = m_X[t.iv1];
		v[2] = m_X[t.iv2];
		v[3] = m_X[t.iv3];
		
		if(pointInsideTetrahedronTest(p, v) ) return i;
	}
	return 0;
}

}