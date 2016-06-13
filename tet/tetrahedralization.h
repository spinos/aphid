/*
 *  tetrahedralization.h
 *  foo
 *
 *  Created by jian zhang on 6/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_TETRAHEDRALIZATION_H
#define TTG_TETRAHEDRALIZATION_H
#include <triangle_math.h>
#include <tetrahedron_math.h>
#include "triangulation.h"

namespace ttg {

struct ITetrahedron {
	
	ITetrahedron * nei0;
	ITetrahedron * nei1;
	ITetrahedron * nei2;
	ITetrahedron * nei3;
	int iv0, iv1, iv2, iv3;
	int index;
};

inline void setTetrahedronVertices(ITetrahedron & t, 
									const int & a, const int & b, 
									const int & c, const int & d)
{ t.iv0 = a; t.iv1 = b; t.iv2 = c; t.iv3 = d; }

inline void resetTetrahedronNeighbors(ITetrahedron & t)
{ t.nei0 = t.nei1 = t.nei2 = t.nei3 = NULL; }

inline void copyTetrahedron(ITetrahedron * a, const ITetrahedron * b)
{
	a->nei0 = b->nei0;
	a->nei1 = b->nei1;
	a->nei2 = b->nei2;
	a->nei3 = b->nei3;
	a->iv0 = b->iv0;
	a->iv1 = b->iv1;
	a->iv2 = b->iv2;
	a->iv3 = b->iv3;
}

inline bool tetrahedronHas6Neighbors(const ITetrahedron * t)
{
    if(!t->nei0) return false;   
    if(!t->nei1) return false; 
    if(!t->nei2) return false;
    if(!t->nei3) return false; 
    return true;
}

inline void printTetrahedronVertices(const ITetrahedron * a)
{ std::cout<<" tetrahedron "<<a->index<<" ("<<a->iv0<<", "<<a->iv1<<", "<<a->iv2<<", "<<a->iv3<<") "; }

inline void printTetrahedronNeighbors(const ITetrahedron * t)
{
    std::cout<<"\n neighbor of "; printTetrahedronVertices(t);
    std::cout<<"\n nei0 "; 
    if(t->nei0) printTetrahedronVertices(t->nei0); 
    std::cout<<"\n nei1 "; 
    if(t->nei1) printTetrahedronVertices(t->nei1); 
    std::cout<<"\n nei2 "; 
    if(t->nei2) printTetrahedronVertices(t->nei2); 
    std::cout<<"\n nei3 "; 
    if(t->nei3) printTetrahedronVertices(t->nei3); 
    
}

inline void faceOfTetrahedron(ITRIANGLE * tri, const ITetrahedron * t, int side)
{
	if(side==0)
		setTriangleVertices(tri, t->iv1, t->iv2, t->iv3);
	else if(side==1)
		setTriangleVertices(tri, t->iv0, t->iv1, t->iv3);
	else if(side==2)
		setTriangleVertices(tri, t->iv0, t->iv2, t->iv1);
	else
		setTriangleVertices(tri, t->iv0, t->iv3, t->iv2);
}

inline int findTetrahedronFace(const ITetrahedron * a, const ITRIANGLE * f)
{
	ITRIANGLE tria;
	int i=0;
	for(;i<4;++i) {
		faceOfTetrahedron(&tria, a, i);
		if(matchTriangles(&tria, f) ) 
			return i;
	}
	return -1;
}

inline ITetrahedron * neighborOfTetrahedron(const ITetrahedron * t,
								const int & a, const int & b, const int &c)
{
	ITRIANGLE trif;
	setTriangleVertices(&trif, a, b, c);
	int side = findTetrahedronFace(t, &trif);
	if(side < 0) return NULL;
	if(side ==0) return t->nei0;
	if(side ==1) return t->nei1;
	if(side ==2) return t->nei2;
	return t->nei3;
}

inline void printTetrahedronHasNoFace(const ITetrahedron * a, const ITRIANGLE * f)
{
	std::cout<<"\n\n [WARNING] no face "; printTriangleVertice(f);
	std::cout<<"\n				in "; printTetrahedronVertices(a);
}

inline bool findSharedFace(int & ia, int & jb,
				const ITetrahedron * a, const ITetrahedron * b)
{
	ia = -1, jb = -1;
	ITRIANGLE tria, trib;
	int i, j;
	for(i=0;i<4;++i) {
		faceOfTetrahedron(&tria, a, i);
		j = findTetrahedronFace(b, &tria);
		if(j > -1 ) {
			ia = i; jb = j;
			return true;
		}
	}
	
	return false;
}

inline void setTetrahedronNeighbor(ITetrahedron * a, ITetrahedron * b, int i)
{
    if(i==0) a->nei0 = b;
    else if(i==1) a->nei1 = b;
    else if(i==2) a->nei2 = b;
    else a->nei3 = b;
}

inline bool connectTetrahedrons(ITetrahedron * a, ITetrahedron * b)
{
	if(!a) return false;
	if(!b) return false;
	
	int i, j, ia, jb;
	for(i=0;i<4;++i) {
		if(findSharedFace(ia, jb, a, b) )
			break;
	}
	
	if(ia > -1 && jb > -1) {
	    setTetrahedronNeighbor(a, b, ia);
	    setTetrahedronNeighbor(b, a, jb);
		//a->nei[ia] = b;
		//b->nei[jb] = a;
		//std::cout<<"\n connect "; printTetrahedronVertices(a);
		//std::cout<<"\n     and "; printTetrahedronVertices(b);
		//std::cout<<std::endl;
		return true;
	}
	
	std::cout<<"\n\n [WARNING] cannot connect "; printTetrahedronVertices(a);
	std::cout<<"\n                        and "; printTetrahedronVertices(b);
	std::cout<<std::endl;
		
	return false;
}

inline void reconnectTetrahedronNeighbors(ITetrahedron * t)
{
    std::cout<<"\n reconnect neighbor of "; printTetrahedronVertices(t);
    if(t->nei0)
        connectTetrahedrons(t, t->nei0); 
    if(t->nei1)
        connectTetrahedrons(t, t->nei1); 
    if(t->nei2)
        connectTetrahedrons(t, t->nei2); 
    if(t->nei3)
        connectTetrahedrons(t, t->nei3); 
}

inline bool checkTetrahedronConnections(ITetrahedron * a)
{
	if(a->nei0) {
		if(!connectTetrahedrons(a, a->nei0) )
			return false;
	}
	if(a->nei1) {
		if(!connectTetrahedrons(a, a->nei1) )
			return false;
	}
	if(a->nei2) {
		if(!connectTetrahedrons(a, a->nei2) )
			return false;
	}
	if(a->nei3) {
		if(!connectTetrahedrons(a, a->nei3) )
			return false;
	}
	return true;
}

inline int oppositeVertex(const ITetrahedron * t, 
						int a, int b, int c)
{
	if(t->iv0!=a && t->iv0 != b && t->iv0 != c) return t->iv0;
	if(t->iv1!=a && t->iv1 != b && t->iv1 != c) return t->iv1;
	if(t->iv2!=a && t->iv2 != b && t->iv2 != c) return t->iv2;
	
	return t->iv3;
}

/// two tetrahedrons sharing same face (1, 2, 3)
/// and six neighbors
typedef struct {
	ITetrahedron * ta;
	ITetrahedron * tb;
	int iv0, iv1, iv2, iv3, iv4;
	
} Bipyramid;

inline void printBipyramidVertices(const Bipyramid * pyra)
{
	std::cout<<" bipyramid ("<<pyra->iv0<<", "
		<<pyra->iv1<<", "<<pyra->iv2<<", "<<pyra->iv3<<", "
		<<pyra->iv4<<") ";
}

inline void resetBipyramid(Bipyramid * pyra)
{ pyra->tb = NULL; }

inline void getNeighborOfBipyramid(ITetrahedron * nei1,
                                ITetrahedron * nei2,
                                ITetrahedron * nei3,
                                ITetrahedron * nei4,
                                ITetrahedron * nei5,
                                ITetrahedron * nei6,
                                const Bipyramid * pyra)
{
    const int v0 = pyra->iv0;
    const int v1 = pyra->iv1;
	const int v2 = pyra->iv2;
	const int v3 = pyra->iv3;
	const int v4 = pyra->iv4;
	nei1 = neighborOfTetrahedron(pyra->ta, v0, v1, v3);
	nei2 = neighborOfTetrahedron(pyra->ta, v0, v2, v1);
	nei3 = neighborOfTetrahedron(pyra->ta, v0, v3, v2);
	nei4 = neighborOfTetrahedron(pyra->tb, v4, v1, v2);
	nei5 = neighborOfTetrahedron(pyra->tb, v4, v2, v3);
	nei6 = neighborOfTetrahedron(pyra->tb, v4, v3, v1);   
}

inline bool createBipyramid(Bipyramid * pyra, 
						ITetrahedron * ta, 
						ITetrahedron * tb)
{
	// printTetrahedronVertices(ta); printTetrahedronVertices(tb);
	pyra->ta = ta;
	pyra->tb = tb;
	int ia, jb;
	if(!findSharedFace(ia, jb, ta, tb) ) {
		std::cout<<"\n [WARNING] not connected ";
		return false;
	}
		
	ITRIANGLE tria;
	faceOfTetrahedron(&tria, ta, ia);
	
	pyra->iv0 = oppositeVertex(ta, tria.p1, tria.p2, tria.p3);
	pyra->iv1 = tria.p1;
	pyra->iv2 = tria.p2;
	pyra->iv3 = tria.p3;
	pyra->iv4 = oppositeVertex(tb, tria.p1, tria.p2, tria.p3);
	
	std::cout<<"\n create "; printBipyramidVertices(pyra);
	if(!checkTetrahedronConnections(ta) ) printTetrahedronNeighbors(ta);
	if(!checkTetrahedronConnections(tb) ) printTetrahedronNeighbors(tb);
	std::cout<<"\n success ";
	return true;
}

/// split t1 to t1 t2 t3 t4 by vi
inline void splitTetrahedron(ITetrahedron * t1, ITetrahedron * t2, 
							ITetrahedron * t3, ITetrahedron * t4,
							int vi)
{
	std::cout<<"\n split "; printTetrahedronVertices(t1);
	printTetrahedronNeighbors(t1);
/// vertices of t1		
	const int v0 = t1->iv0;
	const int v1 = t1->iv1;
	const int v2 = t1->iv2;
	const int v3 = t1->iv3;
	
	ITetrahedron * nei1 = neighborOfTetrahedron(t1, v1, v2, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(t1, v0, v1, v3);
	ITetrahedron * nei3 = neighborOfTetrahedron(t1, v0, v2, v1);
	ITetrahedron * nei4 = neighborOfTetrahedron(t1, v0, v3, v2);
		
	setTetrahedronVertices(*t1, vi, v1, v2, v3);
	setTetrahedronVertices(*t2, vi, v0, v1, v3);
	setTetrahedronVertices(*t3, vi, v0, v2, v1);
	setTetrahedronVertices(*t4, vi, v0, v3, v2);

	connectTetrahedrons(t1, t2);
	connectTetrahedrons(t1, t3);
	connectTetrahedrons(t1, t4);
	connectTetrahedrons(t1, nei1);
	
	connectTetrahedrons(t2, t3);
	connectTetrahedrons(t2, t4);
	connectTetrahedrons(t2, nei2);
	
	connectTetrahedrons(t3, t4);
	connectTetrahedrons(t3, nei3);
	
	connectTetrahedrons(t4, nei4);
	
	std::cout<<"\n to "; printTetrahedronVertices(t1);
	std::cout<<"\n  + "; printTetrahedronVertices(t2);
	std::cout<<"\n  + "; printTetrahedronVertices(t3);
	std::cout<<"\n  + "; printTetrahedronVertices(t4);
	
	if(!checkTetrahedronConnections(t1) ) printTetrahedronNeighbors(t1);
	if(!checkTetrahedronConnections(t2) ) printTetrahedronNeighbors(t2);
	if(!checkTetrahedronConnections(t3) ) printTetrahedronNeighbors(t3);
	if(!checkTetrahedronConnections(t4) ) printTetrahedronNeighbors(t4);
	std::cout<<"\n end of split ";
}

typedef struct {
   aphid::Vector3F pc;
   float r;
} TetSphere;

/// reference http://mathworld.wolfram.com/Circumsphere.html
inline void circumSphere(TetSphere & sphere, 
						const aphid::Vector3F & p1,
						const aphid::Vector3F & p2,
						const aphid::Vector3F & p3,
						const aphid::Vector3F & p4)
{
	aphid::Matrix44F a;
	*a.m(0,0) = p1.x; *a.m(0,1) = p1.y; *a.m(0,2) = p1.z; *a.m(0,3) = 1.f;
	*a.m(1,0) = p2.x; *a.m(1,1) = p2.y; *a.m(1,2) = p2.z; *a.m(1,3) = 1.f;
	*a.m(2,0) = p3.x; *a.m(2,1) = p3.y; *a.m(2,2) = p3.z; *a.m(2,3) = 1.f;
	*a.m(3,0) = p4.x; *a.m(3,1) = p4.y; *a.m(3,2) = p4.z; *a.m(3,3) = 1.f;
	float da = 2.f * a.determinant();
	
	float d1 = p1.length2();
	float d2 = p2.length2();
	float d3 = p3.length2();
	float d4 = p4.length2();
	
	*a.m(0,0) = d1;
	*a.m(1,0) = d2;
	*a.m(2,0) = d3;
	*a.m(3,0) = d4;
	
	float dx = a.determinant();
	
	*a.m(0,1) = p1.x; *a.m(0,2) = p1.z;
	*a.m(1,1) = p2.x; *a.m(1,2) = p2.z;
	*a.m(2,1) = p3.x; *a.m(2,2) = p3.z;
	*a.m(3,1) = p4.x; *a.m(3,2) = p4.z;
	
	float dy = -a.determinant();
	
	*a.m(0,1) = p1.x; *a.m(0,2) = p1.y;
	*a.m(1,1) = p2.x; *a.m(1,2) = p2.y;
	*a.m(2,1) = p3.x; *a.m(2,2) = p3.y;
	*a.m(3,1) = p4.x; *a.m(3,2) = p4.y;
	
	float dz = a.determinant();
	
	sphere.pc.x = dx / da;
	sphere.pc.y = dy / da;
	sphere.pc.z = dz / da;
	sphere.r = p1.distanceTo(sphere.pc);
}

inline bool canSplitFlip(const Bipyramid & pyra, 
							const aphid::Vector3F * X)
{
/// belongs to supertetrahedron
	if(pyra.iv0 < 4 || pyra.iv4 < 4) return false;
	
	if(!tetrahedronHas6Neighbors(pyra.ta) )
		return false;
		
	if(!tetrahedronHas6Neighbors(pyra.tb) )
		return false;
		
/// must be convex
	aphid::Vector3F a = X[pyra.iv1];
	aphid::Vector3F b = X[pyra.iv2];
	aphid::Vector3F c = X[pyra.iv3];
	aphid::Vector3F apex = X[pyra.iv0];
	aphid::Vector3F antipex = X[pyra.iv4];
	if(!aphid::segmentIntersectTriangle(apex, antipex, a, b, c) ) 
		return false;
		
	TetSphere circ;
	circumSphere(circ, apex, a, b, c );
	if(antipex.distanceTo(circ.pc) < circ.r )
		return true;
		
	return false;
}

inline bool canMergeFlip(int & i3rd, const Bipyramid & pyra, 
							const aphid::Vector3F * X)
{
	if(!tetrahedronHas6Neighbors(pyra.ta) )
		return false;
		
	if(!tetrahedronHas6Neighbors(pyra.tb) )
		return false;

	aphid::Vector3F p0 = X[pyra.iv0];
	aphid::Vector3F p1 = X[pyra.iv1];
	aphid::Vector3F p2 = X[pyra.iv2];
	aphid::Vector3F p3 = X[pyra.iv3];
	aphid::Vector3F p4 = X[pyra.iv4];
	TetSphere circ;
	
	int v0 = pyra.iv0;
	int v1 = pyra.iv1;
	int v2 = pyra.iv2;
	int v3 = pyra.iv3;
	int v4 = pyra.iv4;
	
	ITetrahedron * nei1 = neighborOfTetrahedron(pyra.ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(pyra.ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(pyra.ta, v0, v3, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(pyra.tb, v4, v1, v2);
	ITetrahedron * nei5 = neighborOfTetrahedron(pyra.tb, v4, v2, v3);
	ITetrahedron * nei6 = neighborOfTetrahedron(pyra.tb, v4, v3, v1);
	
/// find shared neighbor pair	
	if(nei1 && nei1 == nei6) {
/// be convex
		if(!aphid::segmentIntersectTriangle(p3, p1, p2, p4, p0) )
			return false;
/// empty principle
		circumSphere(circ, p3, p2, p4, p0 );
		if(p1.distanceTo(circ.pc) < circ.r )
			return false;
			
		i3rd = 1;
		return true;
	}
	
	if(nei2 && nei2 == nei4) {
		if(!aphid::segmentIntersectTriangle(p1, p2, p3, p4, p0) )
			return false;
		
		circumSphere(circ, p1, p3, p4, p0 );
		if(p2.distanceTo(circ.pc) < circ.r )
			return false;
		
		i3rd = 2;
		return true;
	}
	
	if(nei3 && nei3 == nei5) {
		if(!aphid::segmentIntersectTriangle(p2, p3, p1, p4, p0) )
			return false;
		
		circumSphere(circ, p2, p1, p4, p0 );
		if(p3.distanceTo(circ.pc) < circ.r )
			return false;
		
		i3rd = 3;
		return true;
	}
	
	return false;
}

inline int tetrahedronHasVertex(const ITetrahedron * t,
								const int & x)
{
	if(t->iv0 == x) return 0;
	if(t->iv1 == x) return 1;
	if(t->iv2 == x) return 2;
	if(t->iv3 == x) return 3;
	return -1;
}

inline bool checkBipyramidConnection(const Bipyramid & pyra)
{
	const int v0 = pyra.iv0;
	const int v1 = pyra.iv1;
	const int v2 = pyra.iv2;
	const int v3 = pyra.iv3;
	const int v4 = pyra.iv4;
	
	ITetrahedron * nei1 = neighborOfTetrahedron(pyra.ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(pyra.ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(pyra.ta, v0, v3, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(pyra.tb, v4, v1, v2);
	ITetrahedron * nei5 = neighborOfTetrahedron(pyra.tb, v4, v2, v3);
	ITetrahedron * nei6 = neighborOfTetrahedron(pyra.tb, v4, v3, v1);
	
	if(tetrahedronHasVertex(nei1, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei2, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei3, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei4, pyra.iv0) > -1 ) return false;
	if(tetrahedronHasVertex(nei5, pyra.iv0) > -1 ) return false;
	if(tetrahedronHasVertex(nei6, pyra.iv0) > -1 ) return false;
	
	return true;
}

/// flip edge by split bipyramid into three tetrahedrons
inline void processSplitFlip(Bipyramid & pyra,
							ITetrahedron * tets,
							int & numTets)
{
	std::cout<<"\n split flip"; printBipyramidVertices(&pyra);
	printTetrahedronVertices(pyra.ta);
	printTetrahedronVertices(pyra.tb);
	// printBipyramidNeighbors(&pyra);
/// vertices of bipyramid		
	const int v0 = pyra.iv0;
	const int v1 = pyra.iv1;
	const int v2 = pyra.iv2;
	const int v3 = pyra.iv3;
	const int v4 = pyra.iv4;
	
/// neighbor of bipyramid
	ITetrahedron * nei1 = neighborOfTetrahedron(pyra.ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(pyra.ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(pyra.ta, v0, v3, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(pyra.tb, v4, v1, v2);
	ITetrahedron * nei5 = neighborOfTetrahedron(pyra.tb, v4, v2, v3);
	ITetrahedron * nei6 = neighborOfTetrahedron(pyra.tb, v4, v3, v1);
	
	if(!checkBipyramidConnection(pyra) ) {
		std::cout<<"\n [ERROR] wrong bipyramid connections";
	}
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tb) ) printTetrahedronNeighbors(pyra.tb);
	
	setTetrahedronVertices(*pyra.ta, v0, v1, v2, v4);
	setTetrahedronVertices(*pyra.tb, v0, v2, v3, v4);

/// add a new one
	ITetrahedron * tc = &tets[numTets]; tc->index = numTets;
	numTets++;
	setTetrahedronVertices(*tc, v0, v1, v4, v3);
	
	connectTetrahedrons(pyra.ta, pyra.tb);
	connectTetrahedrons(pyra.ta, tc);
	connectTetrahedrons(tc, pyra.tb);

	connectTetrahedrons(nei2, pyra.ta);
	connectTetrahedrons(nei4, pyra.ta);
	connectTetrahedrons(nei3, pyra.tb);
	connectTetrahedrons(nei5, pyra.tb);
	connectTetrahedrons(nei1, tc);
	connectTetrahedrons(nei6, tc);
	
	std::cout<<"\n edge ("<<v0<<", "<<v4<<")";
	std::cout<<"\n aft "; printTetrahedronVertices(pyra.ta);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tb);
	std::cout<<"\n   + "; printTetrahedronVertices(tc);
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tb) ) printTetrahedronNeighbors(pyra.tb);
	if(!checkTetrahedronConnections(tc) ) printTetrahedronNeighbors(tc);
}

/// flip edge by merge a bipyramid with a neighbor tetrahedron into two tetrahedrons
inline void processMergeFlip(Bipyramid * pyra,
							const int & side,
							ITetrahedron * tets,
							int & numTets)
{
	std::cout<<"\n merge flip"; printBipyramidVertices(pyra);
	
	ITetrahedron * ta = pyra->ta;
	ITetrahedron * tb = pyra->tb;
	
	const int v0 = pyra->iv0;
	const int v1 = pyra->iv1;
	const int v2 = pyra->iv2;
	const int v3 = pyra->iv3;
	const int v4 = pyra->iv4;
	
	ITetrahedron * nei1 = neighborOfTetrahedron(pyra->ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(pyra->ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(pyra->ta, v0, v3, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(pyra->tb, v4, v1, v2);
	ITetrahedron * nei5 = neighborOfTetrahedron(pyra->tb, v4, v2, v3);
	ITetrahedron * nei6 = neighborOfTetrahedron(pyra->tb, v4, v3, v1);

	ITetrahedron * tc;
	if(side == 1) tc = nei1;
	else if(side == 2) tc = nei2;
	else tc = nei3;
	std::cout<<" and"; printTetrahedronVertices(tc);
	
/// connection to c	
	ITetrahedron * neica;
	ITetrahedron * neicc;
	
	if(side == 1) {
		neica = neighborOfTetrahedron(tc, v0, v1, v4);
		neicc = neighborOfTetrahedron(tc, v0, v4, v3);
		
		setTetrahedronVertices(*ta, v1, v2, v0, v4);
		setTetrahedronVertices(*tc, v3, v4, v0, v2);

		connectTetrahedrons(ta, nei2);
		connectTetrahedrons(ta, nei4);
		connectTetrahedrons(tc, nei3);
		connectTetrahedrons(tc, nei5);
	}
	else if(side == 2) {
		neica = neighborOfTetrahedron(tc, v0, v2, v4);
		neicc = neighborOfTetrahedron(tc, v0, v4, v1);
		
		setTetrahedronVertices(*ta, v2, v3, v0, v4);
		setTetrahedronVertices(*tc, v1, v4, v0, v3);
		
		connectTetrahedrons(ta, nei3);
		connectTetrahedrons(ta, nei5);
		connectTetrahedrons(tc, nei1);
		connectTetrahedrons(tc, nei6);
	}
	else {
		neica = neighborOfTetrahedron(tc, v0, v3, v4);
		neicc = neighborOfTetrahedron(tc, v0, v4, v2);
		
		setTetrahedronVertices(*ta, v3, v1, v0, v4);
		setTetrahedronVertices(*tc, v2, v4, v0, v1);
		
		connectTetrahedrons(ta, nei1);
		connectTetrahedrons(ta, nei6);
		connectTetrahedrons(tc, nei2);
		connectTetrahedrons(tc, nei4);
	}

	connectTetrahedrons(ta, tc);
	connectTetrahedrons(ta, neica);
	connectTetrahedrons(tc, neicc);
	
/// remove tb
	std::cout<<"\n remove "; printTetrahedronVertices(tb); 
	pyra->tb->index = -1;
	
	std::cout<<"\n aft "; printTetrahedronVertices(ta);
	std::cout<<"\n   + "; printTetrahedronVertices(tc);
	
	if(!checkTetrahedronConnections(tc)) {
		std::cout<<"\n [ERROR] wrong tetrahedron connections"; printTetrahedronVertices(tc);
	}
	
	if(!checkTetrahedronConnections(pyra->ta) ) printTetrahedronNeighbors(pyra->ta);
	if(!checkTetrahedronConnections(tc) ) printTetrahedronNeighbors(tc);
}

/// once an face is flipped, spawn three more dipyramids for potential face flipping 
inline void spawnFaces(std::deque<Bipyramid> & pyras,
						ITetrahedron * tc)
{
	Bipyramid p0 = pyras[0];
	std::cout<<"\n spawn "; printBipyramidVertices(&p0);
	
	const int v1 = p0.iv1;
	const int v2 = p0.iv2;
	const int v3 = p0.iv3;
	const int v4 = p0.iv4;
	ITetrahedron * nei4 = neighborOfTetrahedron(p0.ta, v4, v1, v2);
	
	if(nei4) {std::cout<<"\n side 4";
		Bipyramid pa;
		if(createBipyramid(&pa, p0.ta, nei4) )
			pyras.push_back(pa);
	}
	
	ITetrahedron * nei5 = neighborOfTetrahedron(p0.tb, v4, v2, v3);
	
	if(nei5) {std::cout<<"\n side 5";
		Bipyramid pb;
		if(createBipyramid(&pb, p0.tb, nei5) )
			pyras.push_back(pb);
	}
	
	ITetrahedron * nei6 = neighborOfTetrahedron(tc, v4, v3, v1);

	if(nei6) {std::cout<<"\n side 6";
		Bipyramid pc;
		if(createBipyramid(&pc, tc, nei6) )
			pyras.push_back(pc);
	}
}

inline void flipFaces(std::deque<Bipyramid> & pyras, 
							const aphid::Vector3F * X,
							ITetrahedron * tets,
							int & numTets,
							bool recursively = true)
{
	int nq = pyras.size();
	int mergNei, i=0;
	while(nq>0) {
		if(canSplitFlip(pyras[0], X) ) {
			processSplitFlip(pyras[0], tets, numTets);
			if(recursively) spawnFaces(pyras, &tets[numTets-1] );
		}
		else if(canMergeFlip(mergNei, pyras[0], X) ) {
			processMergeFlip(&pyras[0], mergNei, 
							tets, numTets);
		}
		pyras.erase(pyras.begin() );
		nq = pyras.size();
	}
	std::cout<<"\n end flip";
}

}
#endif
