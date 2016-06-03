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
using namespace aphid;
namespace ttg {

Delaunay2D::Delaunay2D() {}
Delaunay2D::~Delaunay2D() 
{
	delete[] m_X;
	delete[] m_triangles;
}

bool Delaunay2D::init() 
{
	std::cout<<" generate samples by 16 X 16 grid\n";
	BoundingBox bbx(0.f, 0.f, 0.f,
					32.f, 32.f, 32.f);
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
			if(RandomF01() < .7f )
				bgg.addCell(sample, level, 0);
		}
	}
	m_N = bgg.numCells();
	std::cout<<" n background cell "<<m_N<<std::endl;
	
/// first three for super triangle
	m_X = new Vector3F[m_N+3];
	m_triangles = new ITRIANGLE[m_N * 3 + 9];
	float dx = bbx.distance(0) + bbx.distance(1);
	m_X[0].set(bbx.getMin(0) - dx * 3.f, bbx.getMin(1) - dx, 0.f);
	m_X[1].set(bbx.getMax(0) + dx, bbx.getMin(1) - dx, 0.f);
	m_X[2].set(bbx.getMax(0) + dx, bbx.getMax(1) + dx * 3.f, 0.f);
	
	i = 3;
	
	sdb::CellHash * cel = bgg.cells();
	cel->begin();
	while(!cel->end()) {
		sample = bgg.cellCenter(cel->key());
		
		m_X[i++].set(sample.x + RandomFn11() * 0.4f, sample.y + RandomFn11() * 0.4f, 0.f);
	    cel->next();
	}
	
	ITRIANGLE superTri;
	superTri.p1 = 0;
	superTri.p2 = 1;
	superTri.p3 = 2;
	m_numTri = 0;
	m_triangles[m_numTri++] = superTri;
	
	i = 3;
#define ENDNV 11
	for(;i<ENDNV;++i) {
/// Lawson's find the triangle that contains X[i]
		j = searchTri(m_X[i]);
		ITRIANGLE t = m_triangles[j];

/// vertices of Tri[j]		
		const int p1 = t.p1;
		const int p2 = t.p2;
		const int p3 = t.p3;
		std::cout<<"\n split tri["<<j<<"]";
		printTriangleVertice(&t);
/// neighbor of Tri[j]
		int quadrinei1, quadrinei2, quadrinei3;
		ITRIANGLE * nei1 = t.nei[0];
		if(nei1) {
			quadrinei1 = findNeighborEdge(nei1, &m_triangles[j]);
		}
		ITRIANGLE * nei2 = t.nei[1];
		if(nei2) {
			quadrinei2 = findNeighborEdge(nei2, &m_triangles[j]);
		}
		ITRIANGLE * nei3 = t.nei[2];
		if(nei3) {
			quadrinei3 = findNeighborEdge(nei3, &m_triangles[j]);
		}

/// remove Tri[j], add three new triangles
/// connect X[i] to be p3		
		m_triangles[j].p3 = i;
		std::cout<<" = ";
		printTriangleVertice(&m_triangles[j]);
		
		t.p1 = p2;
		t.p2 = p3;
		t.p3 = i;
		m_triangles[m_numTri++] = t;
		std::cout<<"  + ";
		printTriangleVertice(&m_triangles[m_numTri-2]);

		t.p1 = p3;
		t.p2 = p1;
		t.p3 = i;
		m_triangles[m_numTri++] = t;
		std::cout<<"  + ";
		printTriangleVertice(&m_triangles[m_numTri-1]);
		
/// connect neighbors to new triangles
		m_triangles[j].nei[1] = &m_triangles[m_numTri-2];
		m_triangles[j].nei[2] = &m_triangles[m_numTri-1];
		
		m_triangles[m_numTri-2].nei[0] = t.nei[1];
		m_triangles[m_numTri-2].nei[1] = &m_triangles[m_numTri-1];
		m_triangles[m_numTri-2].nei[2] = &m_triangles[j];
		
		m_triangles[m_numTri-1].nei[0] = t.nei[2];
		m_triangles[m_numTri-1].nei[1] = &m_triangles[j];
		m_triangles[m_numTri-1].nei[2] = &m_triangles[m_numTri-2];
		
/// add first three potential edges to flip
		std::deque<Quadrilateral> qls;
		Quadrilateral q1;
		q1.ta = &m_triangles[j];
		q1.tb = m_triangles[j].nei[0];
		q1.apex = i;
		findAntiApex(q1);
		qls.push_back(q1);
		
		Quadrilateral q2;
		q2.ta = &m_triangles[m_numTri-2];
		q2.tb = m_triangles[m_numTri-2].nei[0];
		q2.apex = i;
		findAntiApex(q2);
		qls.push_back(q2);
		
		Quadrilateral q3;
		q3.ta = &m_triangles[m_numTri-1];
		q3.tb = m_triangles[m_numTri-1].nei[0];
		q3.apex = i;
		findAntiApex(q3);
		qls.push_back(q3);

/// update new triangles to neighbors
		if(nei2) {
			nei2->nei[quadrinei2] = &m_triangles[m_numTri-2];
		}
		
		if(nei3) {
			nei3->nei[quadrinei3] = &m_triangles[m_numTri-1];
		}
		
		flipEdges(qls);
	}
	return true;
}

const char * Delaunay2D::titleStr() const
{ return "Delaunay 2D Test"; }

void Delaunay2D::draw(GeoDrawer * dr) 
{
	dr->setColor(0.f, 0.f, 0.f);
	int i = 0;
	for(;i<m_N;++i) {
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

/// calculate circumcircle of a triangle in 2D
/// reference http://mathworld.wolfram.com/Circumcircle.html
void Delaunay2D::circumCircle(TriCircle & circ,
						const Vector3F & p1,
						const Vector3F & p2,
						const Vector3F & p3) const
{
/// reference http://mathworld.wolfram.com/Circumradius.html
	//float a = p1.distanceTo(p2);
	//float b = p2.distanceTo(p3);
	//float c = p3.distanceTo(p1);	
	//circ.r = a * b * c / sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) );
	
	Matrix33F ma;
	*ma.m(0, 0) = p1.x; *ma.m(0, 1) = p1.y; *ma.m(0, 2) = 1.f;
	*ma.m(1, 0) = p2.x; *ma.m(1, 1) = p2.y; *ma.m(1, 2) = 1.f;
	*ma.m(2, 0) = p3.x; *ma.m(2, 1) = p3.y; *ma.m(2, 2) = 1.f;
	float alpha = ma.determinant();
	float x12y12 = p1.x * p1.x + p1.y * p1.y;
	float x22y22 = p2.x * p2.x + p2.y * p2.y;
	float x32y32 = p3.x * p3.x + p3.y * p3.y;
	
	*ma.m(0, 0) = x12y12; *ma.m(0, 1) = p1.y; *ma.m(0, 2) = 1.f;
	*ma.m(1, 0) = x22y22; *ma.m(1, 1) = p2.y; *ma.m(1, 2) = 1.f;
	*ma.m(2, 0) = x32y32; *ma.m(2, 1) = p3.y; *ma.m(2, 2) = 1.f;
	float bx = -ma.determinant();
	
	*ma.m(0, 1) = p1.x; 
	*ma.m(1, 1) = p2.x;
	*ma.m(2, 1) = p3.x; 
	float by = ma.determinant();
	circ.pc.x = - bx / 2.f / alpha;
	circ.pc.y = - by / 2.f / alpha;
	circ.pc.z = 0.f;
	
	circ.r = circ.pc.distanceTo(p1);
}

int Delaunay2D::searchTri(const aphid::Vector3F & p) const
{
	Vector3F a, b, c;
	int i=0;
	for(; i<m_numTri; ++i) {
		const ITRIANGLE t = m_triangles[i];
		a = m_X[t.p1];
		b = m_X[t.p2];
		c = m_X[t.p3];
		
		if(insideTri(p, a, b, c) ) return i;
	}
	return 0;
}

bool Delaunay2D::insideTri(const Vector3F & p,
					const Vector3F & a,
					const Vector3F & b,
					const Vector3F & c) const
{
	Vector3F nor = Vector3F::ZAxis;
	
	Vector3F e01 = b - a;
	Vector3F x0 = p - a;
	if(e01.cross(x0).dot(nor) < 0.f) return false;
	
	Vector3F e12 = c - b;
	Vector3F x1 = p - b;
	if(e12.cross(x1).dot(nor) < 0.f) return false;
	
	Vector3F e20 = a - c;
	Vector3F x2 = p - c;
	if(e20.cross(x2).dot(nor) < 0.f) return false;
	
	return true;
}

void Delaunay2D::flipEdges(std::deque<Quadrilateral> & qls)
{
	int nq = qls.size();
	int i=0;
	while(nq>0) {
		std::cout<<"\n precess quadrilateral["<<i++<<"]\n";
		
		if(canEdgeFlip(qls[0]) ) {
			processEdgeFlip(qls[0]);
		}
		qls.erase(qls.begin() );
		nq--;
	}
}

bool Delaunay2D::canEdgeFlip(const Quadrilateral & q) const
{
	ITRIANGLE * ta = q.ta;
	std::cout<<" edge ("<<q.e.v[0]<<", "<<q.e.v[1]<<")\n";
	std::cout<<" triangle1 ("<<ta->p1<<", "<<ta->p2<<", "<<ta->p3<<")\n"
		<<" apex "<<q.apex<<"\n";
	
	ITRIANGLE * tb = q.tb;
	if(!tb) return false;
	
	std::cout<<" triangle2 ("<<tb->p1<<", "<<tb->p2<<", "<<tb->p3<<")\n";
	std::cout<<" anti-apex "<<q.aapex<<"\n";
/// belongs to supertriangle
	if(q.apex < 3 || q.aapex < 3) return false;
	
	std::cout<<" check empty principle\n";
	TriCircle cir;
	circumCircle(cir, m_X[tb->p1], m_X[tb->p2], m_X[tb->p3]);
	if(m_X[q.apex].distanceTo(cir.pc) <= cir.r )
		return true;
	circumCircle(cir, m_X[ta->p1], m_X[ta->p2], m_X[ta->p3]);
	if(m_X[q.aapex].distanceTo(cir.pc) <= cir.r )
		return true;
	return false;
}

IEDGE Delaunay2D::findEdge(const ITRIANGLE * tri, const int & p) const
{
	IEDGE e;
	int n = 0;
	if(tri->p1 != p) e.v[n++] = tri->p1;
	if(tri->p2 != p) e.v[n++] = tri->p2;
	if(tri->p3 != p) e.v[n++] = tri->p3;
	return e;
}

int Delaunay2D::findAntiApex(const ITRIANGLE * tri, const IEDGE & e) const
{
	if(tri->p1 != e.v[0] && tri->p1 != e.v[1]) return tri->p1;
	if(tri->p2 != e.v[0] && tri->p2 != e.v[1]) return tri->p2;
	return tri->p3;
}

bool Delaunay2D::findAntiApex(Quadrilateral & q) const
{
	ITRIANGLE * tb = q.tb;
	if(!tb) {
		q.aapex = -1;
		return false;
	}
	
	ITRIANGLE * ta = q.ta;
	q.e = findEdge(ta, q.apex);
	q.aapex = findAntiApex(tb, q.e);
	return true;
}

int Delaunay2D::findNeighborEdge(const ITRIANGLE * tri,  ITRIANGLE * tgt) const
{
	if(tri->nei[0] == tgt) return 0;
	if(tri->nei[1] == tgt) return 1;
	return 2;
}

void Delaunay2D::findQuadNeighbor(Quadrilateral & q)
{
	int apex = q.apex;
	if(q.ta->p1 == apex) {
		q.nei[0] = q.ta->nei[2];
		q.nei[1] = q.ta->nei[0];
	}
	else if(q.ta->p2 == apex) {
		q.nei[0] = q.ta->nei[0];
		q.nei[1] = q.ta->nei[1];
	}
	else {
		q.nei[0] = q.ta->nei[1];
		q.nei[1] = q.ta->nei[2];
	}
	apex = q.aapex;
	if(q.tb->p1 == apex) {
		q.nei[2] = q.tb->nei[2];
		q.nei[3] = q.tb->nei[0];
	}
	else if(q.tb->p2 == apex) {
		q.nei[2] = q.tb->nei[0];
		q.nei[3] = q.tb->nei[1];
	}
	else {
		q.nei[2] = q.tb->nei[1];
		q.nei[3] = q.tb->nei[2];
	}
}

void Delaunay2D::processEdgeFlip(Quadrilateral & q)
{
	std::cout<<"\n flip "<<q.e.v[0]<<", "<<q.e.v[1]<<", "<<q.apex<<", "<<q.aapex<<"\n";
	findQuadNeighbor(q);
	int i=0;
	for(;i<4;++i) {
		std::cout<<" neighbor triangle["<<i<<"] ";
		printTriangleVertice(q.nei[i]);
	}
	q.ta->p1 = q.apex;
	q.ta->p2 = q.aapex;
	q.ta->p3 = q.e.v[1];
	q.ta->nei[0] = q.tb;
	q.ta->nei[1] = q.nei[3];
	q.ta->nei[2] = q.nei[0];
	
	q.tb->p1 = q.aapex;
	q.tb->p2 = q.apex;
	q.tb->p3 = q.e.v[0];
	q.tb->nei[0] = q.ta;
	q.tb->nei[1] = q.nei[1];
	q.tb->nei[2] = q.nei[2];
/// update neighbors
	connectTriangleAB(q.nei[0], q.ta, q.apex);
	connectTriangleAB(q.nei[3], q.ta, q.e.v[1]);
	connectTriangleAB(q.nei[1], q.tb, q.e.v[0]);
	connectTriangleAB(q.nei[2], q.tb, q.aapex);
}

bool Delaunay2D::connectTriangleAB(ITRIANGLE * b, ITRIANGLE * a, const int & p)
{
	if(p == b->p1) {
		b->nei[0] = a;
		return true;
	}
	if(p == b->p2) {
		b->nei[1] = a;
		return true;
	}
	if(p == b->p3) {
		b->nei[2] = a;
		return true;
	}
	return false;
}

void Delaunay2D::printTriangleVertice(const ITRIANGLE * a) const
{ std::cout<<" triangle ("<<a->p1<<", "<<a->p2<<", "<<a->p3<<")\n"; }

}