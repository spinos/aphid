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
m_endTet(243) 
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
	m_tets = new ITetrahedron[m_N * 10 + 16];
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
		sample.x += RandomFn11() * 0.33f * hh;
		sample.y += RandomFn11() * 0.33f * hh;
		sample.z += RandomFn11() * 0.33f * hh;
		
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
	superTet.index = 0;
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
		ITetrahedron * t = &m_tets[j];

/// neighbor of t
		ITetrahedron * nei1 = t->nei0;		
		ITetrahedron * nei2 = t->nei1;
		ITetrahedron * nei3 = t->nei2;
		ITetrahedron * nei4 = t->nei3;
		
		ITetrahedron * t1 = &m_tets[j];
		ITetrahedron * t2 = &m_tets[m_numTet]; t2->index = m_numTet;
		ITetrahedron * t3 = &m_tets[m_numTet+1]; t3->index = m_numTet+1;
		ITetrahedron * t4 = &m_tets[m_numTet+2]; t4->index = m_numTet+2;
		
		splitTetrahedron(t1, t2, t3, t4, ii);
/// add three tetrahedrons
		m_numTet+=3;	
		
		std::deque<Bipyramid> pyras;
		
		resetBipyramid(&m_pyra1);
		resetBipyramid(&m_pyra2);
		resetBipyramid(&m_pyra3);
		resetBipyramid(&m_pyra4);
		
		int oldNt;
		
/// update new tetrahedrons to old neighbors
		if(nei1) {
			if(createBipyramid(&m_pyra1, t1, nei1) ) {
				pyras.push_back(m_pyra1);
				oldNt = m_numTet;
				flipFaces(pyras, m_X, m_tets, m_numTet);
				if(oldNt == m_numTet)
					resetBipyramid(&m_pyra1);
			}
		}
		
		if(nei2) {
			if(createBipyramid(&m_pyra2, t2, nei2) ) {
				pyras.push_back(m_pyra2);
				oldNt = m_numTet;
				flipFaces(pyras, m_X, m_tets, m_numTet);
				if(oldNt == m_numTet)
					resetBipyramid(&m_pyra2);
			}
		}
		
		if(nei3) {
			if(createBipyramid(&m_pyra3, t3, nei3) ) {
				pyras.push_back(m_pyra3);
				oldNt = m_numTet;
				flipFaces(pyras, m_X, m_tets, m_numTet);
				if(oldNt == m_numTet)
					resetBipyramid(&m_pyra3);
			}
		}
			
		if(nei4) {
			if(createBipyramid(&m_pyra4, t4, nei4) ) {
				pyras.push_back(m_pyra4);
				oldNt = m_numTet;
				flipFaces(pyras, m_X, m_tets, m_numTet);
				if(oldNt == m_numTet)
					resetBipyramid(&m_pyra4);
			}
		}
		
	}
	if(!checkConnectivity() )
		std::cout<<"\n [ERROR] wrong connectivity ";
	std::cout<<"\n end triangulate X["<<i-1<<"] nt "<<m_numTet;
	std::cout.flush();
	return true;
}

bool Delaunay3D::checkConnectivity()
{
	int i = 0;
	for(;i<m_numTet;++i) {
		if(m_tets[i].index < 0) continue;
		if(!checkTetrahedronConnections(&m_tets[i]) )
			return false;
	}
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
	for(;i<m_endTet;++i) {
		dr->cube(m_X[m_ind[i].value], .125f);
	}
	dr->m_wireProfile.apply();
	dr->setColor(0.2f, 0.2f, 0.4f);
	glBegin(GL_TRIANGLES);
	Vector3F a, b, c, d;
	i = 0;
	for(;i<m_numTet;++i) {
		const ITetrahedron t = m_tets[i];
		if(t.index < 0) continue;
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

#if 0
	dr->setColor(0.92f, 0.2f, 0.f);
	a = m_X[10];
		b = m_X[7];
		c = m_X[0];
		d = m_X[4]; 
	glBegin(GL_TRIANGLES);
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
	glEnd();
#endif

	dr->m_markerProfile.apply();
	
#if 0
	glColor3f(0.7f, .5f, 0.f);
	if(m_pyra1.tb) drawBipyramid(m_pyra1);
	glColor3f(0.f, .5f, .7f);
	//if(m_pyra2.tb) drawBipyramid(m_pyra2);
	glColor3f(.5f, 0.f, .7f);
	if(m_pyra3.tb) drawBipyramid(m_pyra3);
	glColor3f(.5f, .5f, .5f);
	//if(m_pyra4.tb) drawBipyramid(m_pyra4);
#endif

#if 0
	dr->setColor(.5f, .5f, .5f);
	TetSphere cir;
	i = 0;
	for(;i<m_numTet;++i) {
		const ITetrahedron t = m_tets[i];
		if(t.index < 0) continue;
		a = m_X[t.iv0];
		b = m_X[t.iv1];
		c = m_X[t.iv2];
		d = m_X[t.iv3];
		
		circumSphere(cir, a, b, c, d);
		dr->alignedCircle(cir.pc, cir.r);
	}
#endif
}

void Delaunay3D::drawBipyramid(const Bipyramid & pyra) const
{
	Vector3F a, b, c, d, e;
	glBegin(GL_LINES);
	a = m_X[pyra.iv0];
	b = m_X[pyra.iv1];
	c = m_X[pyra.iv2];
	d = m_X[pyra.iv3];
	e = m_X[pyra.iv4];
	
	glVertex3fv((const GLfloat *)&a);
	glVertex3fv((const GLfloat *)&b);
	glVertex3fv((const GLfloat *)&a);
	glVertex3fv((const GLfloat *)&c);
	glVertex3fv((const GLfloat *)&a);
	glVertex3fv((const GLfloat *)&d);
	
	glColor3f(.1f, .1f, .1f);
	glVertex3fv((const GLfloat *)&b);
	glVertex3fv((const GLfloat *)&e);
	glVertex3fv((const GLfloat *)&c);
	glVertex3fv((const GLfloat *)&e);
	glVertex3fv((const GLfloat *)&d);
	glVertex3fv((const GLfloat *)&e);
	
	glEnd();
}

int Delaunay3D::searchTet(const aphid::Vector3F & p) const
{
	Vector3F v[4];
	int i=m_numTet-1;
	for(; i>=0; --i) {
		const ITetrahedron t = m_tets[i];
		if(t.index < 0) continue;
		v[0] = m_X[t.iv0];
		v[1] = m_X[t.iv1];
		v[2] = m_X[t.iv2];
		v[3] = m_X[t.iv3];
		
		if(pointInsideTetrahedronTest(p, v) ) return i;
	}
	return 0;
}

}