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

Delaunay2D::Delaunay2D() :
m_endTri(25) 
{}

Delaunay2D::~Delaunay2D() 
{
	delete[] m_X;
	delete[] m_triangles;
}

bool Delaunay2D::init() 
{
	generateSamples();
	triangulate();
}

void Delaunay2D::generateSamples()
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
			if(RandomF01() < .99f )
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
		int j = searchTri(m_X[i]);
		ITRIANGLE t = m_triangles[j];

/// vertices of Tri[j]		
		const int p1 = t.p1;
		const int p2 = t.p2;
		const int p3 = t.p3;
		std::cout<<"\n split tri["<<j<<"]";
		printTriangleVertice(&t);
/// neighbor of Tri[j]
		ITRIANGLE * nei1 = t.nei[0];		
		ITRIANGLE * nei2 = t.nei[1];
		ITRIANGLE * nei3 = t.nei[2];		

/// remove Tri[j], add three new triangles
/// connect X[i] to be p3		
		m_triangles[j].p3 = i;
		std::cout<<" = ";
		printTriangleVertice(&m_triangles[j]);
		
		t.p1 = p2;
		t.p2 = p3;
		t.p3 = i;
		m_triangles[m_numTri++] = t;
		std::cout<<" + ";
		printTriangleVertice(&m_triangles[m_numTri-1]);

		t.p1 = p3;
		t.p2 = p1;
		t.p3 = i;
		m_triangles[m_numTri++] = t;
		std::cout<<" + ";
		printTriangleVertice(&m_triangles[m_numTri-1]);
		
/// connect neighbors to new triangles
		int ae, be;
		connectTriangles(&m_triangles[j], &m_triangles[m_numTri-2], be, ae);
		connectTriangles(&m_triangles[j], &m_triangles[m_numTri-1], be, ae);
		connectTriangles(&m_triangles[m_numTri-1], &m_triangles[m_numTri-2], be, ae);

/// update new triangles to neighbors
		if(nei1)
			connectTriangles(nei1, &m_triangles[j], be, ae);
			
		if(nei2)
			connectTriangles(nei2, &m_triangles[m_numTri-2], be, ae);
		
		if(nei3)
			connectTriangles(nei3, &m_triangles[m_numTri-1], be, ae);
				
/// add first three potential edges to flip
		std::deque<Quadrilateral> qls;
		Quadrilateral q1;
		q1.ta = &m_triangles[j];
		q1.tb = m_triangles[j].nei[0];
		q1.apex = i;
		findAntiApex(q1);
		qls.push_back(q1);
		flipEdges(qls);
		
		Quadrilateral q2;
		q2.ta = &m_triangles[m_numTri-2];
		q2.tb = m_triangles[m_numTri-2].nei[0];
		q2.apex = i;
		findAntiApex(q2);
		qls.push_back(q2);
		flipEdges(qls);
		
		Quadrilateral q3;
		q3.ta = &m_triangles[m_numTri-1];
		q3.tb = m_triangles[m_numTri-1].nei[0];
		q3.apex = i;
		findAntiApex(q3);
		qls.push_back(q3);
		flipEdges(qls);
		
	}
	std::cout<<" end triangulate X["<<i-1<<"]"<<std::endl;
	return true;
}

bool Delaunay2D::progressForward()
{ 
	m_endTri++;
	if(m_endTri>m_N) m_endTri=m_N;
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
		std::cout<<"\n process quadrilateral["<<i++<<"]\n";
		
		if(canEdgeFlip(qls[0]) ) {
			processEdgeFlip(qls[0]);
			spawnEdges(qls);
		}
		qls.erase(qls.begin() );
		nq = qls.size();
	}
}

bool Delaunay2D::canEdgeFlip(const Quadrilateral & q) const
{
	ITRIANGLE * ta = q.ta;
	std::cout<<" ta"; printTriangleVertice(ta);
	std::cout<<" apex "<<q.apex<<"\n";
	
	ITRIANGLE * tb = q.tb;
	if(!tb) return false;
	
	std::cout<<" tb"; printTriangleVertice(tb);
	std::cout<<" anti-apex "<<q.aapex<<"\n";
	std::cout<<" edge ("<<q.e.v[0]<<", "<<q.e.v[1]<<")\n";
	
/// belongs to supertriangle
	if(q.apex < 3 || q.aapex < 3) return false;
	
	std::cout<<" check empty principle\n";
	TriCircle cir;
	circumCircle(cir, m_X[tb->p1], m_X[tb->p2], m_X[tb->p3]);
	if(m_X[q.apex].distanceTo(cir.pc) < cir.r ) {
		//TriCircle cir1;
		//circumCircle(cir1, m_X[q.apex], m_X[q.e.v[1]], m_X[q.aapex]);
		//if(m_X[q.e.v[0]].distanceTo(cir1.pc) <= cir1.r) {
		//	if(cir1.r >= cir.r) return false;
		//}
		return true;
	}
	//circumCircle(cir, m_X[ta->p1], m_X[ta->p2], m_X[ta->p3]);
	//if(m_X[q.aapex].distanceTo(cir.pc) <= cir.r )
	//	return true;
	std::cout<<" empty\n";
	return false;
}

IEDGE Delaunay2D::findOppositeEdge(const ITRIANGLE * tri, const int & p) const
{
	IEDGE e;
	if(tri->p1 == p) {
		e.v[0] = tri->p2;
		e.v[1] = tri->p3;
		return e;
	}
	if(tri->p2 == p) {
		e.v[0] = tri->p3;
		e.v[1] = tri->p1;
		return e;
	}
	if(tri->p3 == p) {
		e.v[0] = tri->p1;
		e.v[1] = tri->p2;
		return e;
	}
	std::cout<<" cannot find opposite edge to "<<p<<" in "; printTriangleVertice(tri);
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
	q.e = findOppositeEdge(ta, q.apex);
	q.aapex = findAntiApex(tb, q.e);
	return true;
}

void Delaunay2D::findQuadNeighbor(Quadrilateral & q)
{
	int apex = q.apex;
	if(q.ta->p1 == apex) {
		q.nei[0] = q.ta->nei[0];
		q.nei[3] = q.ta->nei[2];
	}
	else if(q.ta->p2 == apex) {
		q.nei[0] = q.ta->nei[1];
		q.nei[3] = q.ta->nei[0];
	}
	else {
		q.nei[0] = q.ta->nei[2];
		q.nei[3] = q.ta->nei[1];
	}
	apex = q.aapex;
	if(q.tb->p1 == apex) {
		q.nei[2] = q.tb->nei[0];
		q.nei[1] = q.tb->nei[2];
	}
	else if(q.tb->p2 == apex) {
		q.nei[2] = q.tb->nei[1];
		q.nei[1] = q.tb->nei[0];
	}
	else {
		q.nei[2] = q.tb->nei[2];
		q.nei[1] = q.tb->nei[1];
	}
}

void Delaunay2D::processEdgeFlip(Quadrilateral & q)
{
	std::cout<<"\n flip ";printQuadrilateral(&q);
	findQuadNeighbor(q);
	int i=0;
	for(;i<4;++i) {
		std::cout<<" qudrilateral neighbor["<<i<<"] ";
		printTriangleVertice(q.nei[i]);
	}
	q.ta->p1 = q.apex;
	q.ta->p2 = q.aapex;
	q.ta->p3 = q.e.v[1];
	q.tb->p1 = q.aapex;
	q.tb->p2 = q.apex;
	q.tb->p3 = q.e.v[0];
/// update neighbors
	int ae, be;
	connectTriangles(q.tb, q.ta, be, ae);
	connectTriangles(q.nei[2], q.ta, be, ae);
	connectTriangles(q.nei[3], q.ta, be, ae);
	connectTriangles(q.nei[0], q.tb, be, ae);
	connectTriangles(q.nei[1], q.tb, be, ae);
}

void Delaunay2D::spawnEdges(std::deque<Quadrilateral> & qls)
{
	Quadrilateral q0 = qls[0];
	std::cout<<"\n spawn ";printQuadrilateral(&q0);
	
	std::cout<<" spawn 0";
	Quadrilateral q1;
	q1.ta = q0.ta;
	q1.tb = q0.nei[2];
	q1.apex = q0.apex;
	findAntiApex(q1);
	std::cout<<" spawn quad0"; printQuadrilateral(&q1);
	qls.push_back(q1);
	
	std::cout<<" spawn 1";
	Quadrilateral q2;
	q2.ta = q0.nei[1];
	q2.tb = q0.tb;
	q2.apex = oppositeVertex(q0.nei[1], q0.aapex, q0.e.v[0]);
	findAntiApex(q2);
	std::cout<<" spawn quad1"; printQuadrilateral(&q2);
	qls.push_back(q2);
}

int Delaunay2D::containtsVertex(const ITRIANGLE * a, const int & p) const
{
	if(a->p1 == p) return 0;
	if(a->p2 == p) return 1;
	if(a->p3 == p) return 2;
	return -1;
}

int Delaunay2D::previousVertex(const ITRIANGLE * a, const int & i) const
{
	if(i==0) return a->p3;
	if(i==1) return a->p1;
	return a->p2;
}

int Delaunay2D::currentVertex(const ITRIANGLE * a, const int & i) const
{
	if(i==0) return a->p1;
	if(i==1) return a->p2;
	return a->p3;
}

int Delaunay2D::nextVertex(const ITRIANGLE * a, const int & i) const
{
	if(i==0) return a->p2;
	if(i==1) return a->p3;
	return a->p1;
}

int Delaunay2D::oppositeVertex(const ITRIANGLE * a, const int & va, const int & vb) const
{
	std::cout<<" try find opposite vertex to "<<va<<","<<vb<<" in"; printTriangleVertice(a);
	
	if(a->p1 != va && a->p1 != vb) return a->p1;
	if(a->p2 != va && a->p2 != vb) return a->p2;
	if(a->p3 != va && a->p3 != vb) return a->p3;
	std::cout<<" cannot find opposite vertex to "<<va<<","<<vb<<" in"; printTriangleVertice(a);
	return -1;
}

bool Delaunay2D::connectTriangles(ITRIANGLE * b, ITRIANGLE * a,
								int & be, int & ae) const
{
	for(int i=0; i<3; ++i) {
		ae = i;
		be = containtsVertex(b, currentVertex(a, ae) );
		if(be > -1) {
			if(nextVertex(a, ae) == previousVertex(b, be) ) {

				a->nei[ae] = b;
				be--;
				if(be<0) be=2;
				b->nei[be] = a;
				
				std::cout<<" connect "; printTriangleVertice(a);
				std::cout<<"     and "; printTriangleVertice(b);
					
				return true;
			}
		}
	}
	
	std::cout<<" cannot connect "; printTriangleVertice(a);
	std::cout<<"            and "; printTriangleVertice(b);
	return false;
}

void Delaunay2D::printTriangleVertice(const ITRIANGLE * a) const
{ std::cout<<" triangle ("<<a->p1<<", "<<a->p2<<", "<<a->p3<<")\n"; }

void Delaunay2D::printQuadrilateral(const Quadrilateral * q) const
{ std::cout<<" quadrilateral ("<<q->apex<<", "<<q->e.v[0]<<", "<<q->aapex<<", "<<q->e.v[1]<<")\n"; }

}