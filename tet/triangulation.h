/*
 *  triangulation.h
 *  foo
 *
 *  Created by jian zhang on 6/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_TRIANGULATION_H
#define TTG_TRIANGULATION_H
#include <AllMath.h>
#include <deque>

namespace ttg {

struct ITRIANGLE {
	ITRIANGLE() {
		nei[0] = nei[1] = nei[2] = NULL;
	}
	
	ITRIANGLE * nei[3];
	int p1,p2,p3;
};

typedef struct {
   int v[2];
} IEDGE;

typedef struct {
   aphid::Vector3F pc;
   float r;
} TriCircle;

inline void printTriangleVertice(const ITRIANGLE * a)
{ std::cout<<" triangle ("<<a->p1<<", "<<a->p2<<", "<<a->p3<<")\n"; }

inline void setTriangleVertices(ITRIANGLE * t, int a, int b, int c)
{ t->p1 = a; t->p2 = b; t->p3 = c; }

inline int containtsVertex(const ITRIANGLE * a, const int & p)
{
	if(a->p1 == p) return 0;
	if(a->p2 == p) return 1;
	if(a->p3 == p) return 2;
	return -1;
}

inline bool matchTriangles(const ITRIANGLE * a, const ITRIANGLE * b)
{
	if(containtsVertex(a, b->p1) < 0) return false;
	if(containtsVertex(a, b->p2) < 0) return false;
	if(containtsVertex(a, b->p3) < 0) return false;
	return true;
}

inline int previousVertex(const ITRIANGLE * a, const int & i)
{
	if(i==0) return a->p3;
	if(i==1) return a->p1;
	return a->p2;
}

inline int currentVertex(const ITRIANGLE * a, const int & i)
{
	if(i==0) return a->p1;
	if(i==1) return a->p2;
	return a->p3;
}

inline int nextVertex(const ITRIANGLE * a, const int & i)
{
	if(i==0) return a->p2;
	if(i==1) return a->p3;
	return a->p1;
}

inline int oppositeVertex(const ITRIANGLE * a, const int & va, const int & vb)
{
	if(a->p1 != va && a->p1 != vb) return a->p1;
	if(a->p2 != va && a->p2 != vb) return a->p2;
	if(a->p3 != va && a->p3 != vb) return a->p3;
	std::cout<<" failed to find opposite vertex to "<<va<<","<<vb<<" in"; printTriangleVertice(a);
	return -1;
}
	
inline bool connectTriangles(ITRIANGLE * b, ITRIANGLE * a,
								int & be, int & ae)
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
	
	std::cout<<" failed to connect "; printTriangleVertice(a);
	std::cout<<"               and "; printTriangleVertice(b);
	return false;
}

inline IEDGE findOppositeEdge(const ITRIANGLE * tri, const int & p)
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
	std::cout<<" failed to find opposite edge to "<<p<<" in "; printTriangleVertice(tri);
	return e;
}

/// calculate circumcircle of a triangle in 2D
/// reference http://mathworld.wolfram.com/Circumcircle.html
inline void circumCircle(TriCircle & circ,
						const aphid::Vector3F & p1,
						const aphid::Vector3F & p2,
						const aphid::Vector3F & p3)
{
/// reference http://mathworld.wolfram.com/Circumradius.html
	//float a = p1.distanceTo(p2);
	//float b = p2.distanceTo(p3);
	//float c = p3.distanceTo(p1);	
	//circ.r = a * b * c / sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) );
	
	aphid::Matrix33F ma;
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

inline bool insideTri(const aphid::Vector3F & p,
					const aphid::Vector3F & a,
					const aphid::Vector3F & b,
					const aphid::Vector3F & c)
{
	aphid::Vector3F nor = aphid::Vector3F::ZAxis;
	
	aphid::Vector3F e01 = b - a;
	aphid::Vector3F x0 = p - a;
	if(e01.cross(x0).dot(nor) < 0.f) return false;
	
	aphid::Vector3F e12 = c - b;
	aphid::Vector3F x1 = p - b;
	if(e12.cross(x1).dot(nor) < 0.f) return false;
	
	aphid::Vector3F e20 = a - c;
	aphid::Vector3F x2 = p - c;
	if(e20.cross(x2).dot(nor) < 0.f) return false;
	
	return true;
}

/// split t1 into t1 t2 t3 by v0
/// with connections
inline void splitTriangle(ITRIANGLE * t1, ITRIANGLE * t2, ITRIANGLE * t3,
			int v0)
{
/// vertices of Tri[j]		
	const int v1 = t1->p1;
	const int v2 = t1->p2;
	const int v3 = t1->p3;
	
/// neighbor of Tri[j]
	ITRIANGLE * nei1 = t1->nei[0];		
	ITRIANGLE * nei2 = t1->nei[1];
	ITRIANGLE * nei3 = t1->nei[2];
		
/// x0 be p3
	t1->p3 = v0;
	
	t2->p1 = v2;
	t2->p2 = v3;
	t2->p3 = v0;
	
	t3->p1 = v3;
	t3->p2 = v1;
	t3->p3 = v0;
	
	int ae, be;
	connectTriangles(t1, t2, be, ae);
	connectTriangles(t1, t3, be, ae);
	connectTriangles(t3, t2, be, ae);

/// update new triangles to neighbors
	if(nei1)
		connectTriangles(nei1, t1, be, ae);
		
	if(nei2)
		connectTriangles(nei2, t2, be, ae);
	
	if(nei3)
		connectTriangles(nei3, t3, be, ae);
}

struct Quadrilateral {
	ITRIANGLE * ta;
	ITRIANGLE * tb;
	ITRIANGLE * nei[4];
	IEDGE e;
	int apex, aapex;
};

inline void printQuadrilateral(const Quadrilateral * q)
{ std::cout<<" quadrilateral ("<<q->apex<<", "<<q->e.v[0]<<", "<<q->aapex<<", "<<q->e.v[1]<<")\n"; }

inline bool findAntiApex(Quadrilateral & q)
{
	ITRIANGLE * tb = q.tb;
	if(!tb) {
		q.aapex = -1;
		return false;
	}
	
	ITRIANGLE * ta = q.ta;
	q.e = findOppositeEdge(ta, q.apex);
	q.aapex = oppositeVertex(tb, q.e.v[0], q.e.v[1] );
	return true;
}

inline void findQuadNeighbor(Quadrilateral & q)
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

inline void processEdgeFlip(Quadrilateral & q)
{
	std::cout<<"\n flip ";printQuadrilateral(&q);
	findQuadNeighbor(q);
	int i=0;
	for(;i<4;++i) {
		std::cout<<" qudrilateral neighbor["<<i<<"] ";
		printTriangleVertice(q.nei[i]);
	}
	q.ta->p1 = q.apex;
	q.ta->p2 = q.e.v[0];
	q.ta->p3 = q.aapex;
	
	q.tb->p1 = q.apex;
	q.tb->p2 = q.aapex;
	q.tb->p3 = q.e.v[1];
/// update neighbors
	int ae, be;
	connectTriangles(q.tb, q.ta, be, ae);
	connectTriangles(q.nei[0], q.ta, be, ae);
	connectTriangles(q.nei[1], q.ta, be, ae);
	connectTriangles(q.nei[2], q.tb, be, ae);
	connectTriangles(q.nei[3], q.tb, be, ae);
}

/// once an edge is flipped, spawn two more quadrilaterals for potential edge flipping 
inline void spawnEdges(std::deque<Quadrilateral> & qls)
{
	Quadrilateral q0 = qls[0];
	std::cout<<"\n spawn ";printQuadrilateral(&q0);
	
	Quadrilateral q1;
	q1.ta = q0.ta;
	q1.tb = q0.nei[1];
	q1.apex = q0.apex;
	findAntiApex(q1);
	std::cout<<" quad0"; printQuadrilateral(&q1);
	qls.push_back(q1);
	
	Quadrilateral q2;
	q2.ta = q0.tb;
	q2.tb = q0.nei[2];
	q2.apex = q0.apex;
	findAntiApex(q2);
	std::cout<<" quad1"; printQuadrilateral(&q2);
	qls.push_back(q2);
}

inline bool canEdgeFlip(const Quadrilateral & q, 
							const aphid::Vector3F * X)
{
	ITRIANGLE * tb = q.tb;
	if(!tb) return false;

	ITRIANGLE * ta = q.ta;
	std::cout<<" ta"; printTriangleVertice(ta);
	//std::cout<<" apex "<<q.apex<<"\n";
	std::cout<<" tb"; printTriangleVertice(tb);
	//std::cout<<" anti-apex "<<q.aapex<<"\n";
	//std::cout<<" edge ("<<q.e.v[0]<<", "<<q.e.v[1]<<")\n";
	
/// belongs to supertriangle
	if(q.apex < 3 || q.aapex < 3) return false;
	
	//std::cout<<" check empty principle\n";
	TriCircle cir;
	circumCircle(cir, X[tb->p1], X[tb->p2], X[tb->p3]);
	if(X[q.apex].distanceTo(cir.pc) < cir.r )
		return true;
	
	std::cout<<" empty\n";
	return false;
}

/// Lawson's algorithm
inline void flipEdges(std::deque<Quadrilateral> & qls, 
							const aphid::Vector3F * X)
{
	int nq = qls.size();
	int i=0;
	while(nq>0) {
		std::cout<<"\n process quadrilateral["<<i++<<"]\n";
		
		if(canEdgeFlip(qls[0], X) ) {
			processEdgeFlip(qls[0]);
			spawnEdges(qls);
		}
		qls.erase(qls.begin() );
		nq = qls.size();
	}
}

}
#endif
