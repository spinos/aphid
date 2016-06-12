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
	
	ITetrahedron * nei[4];
	int iv0, iv1, iv2, iv3;
	
};

inline void setTetrahedronVertices(ITetrahedron & t, 
									const int & a, const int & b, 
									const int & c, const int & d)
{ t.iv0 = a; t.iv1 = b; t.iv2 = c; t.iv3 = d; }

inline void resetTetrahedronNeighbors(ITetrahedron & t)
{ t.nei[0] = t.nei[1] = t.nei[2] = t.nei[3] = NULL; }

inline void printTetrahedronVertices(const ITetrahedron * a)
{ std::cout<<" tetrahedron ("<<a->iv0<<", "<<a->iv1<<", "<<a->iv2<<", "<<a->iv3<<") "; }

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
	return t->nei[side];
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

inline bool connectTetrahedrons(ITetrahedron * a, ITetrahedron * b)
{
	if(!b) return false;
	
	int i, j, ia, jb;
	for(i=0;i<4;++i) {
		if(findSharedFace(ia, jb, a, b) )
			break;
	}
	
	if(ia > -1 && jb > -1) {
		a->nei[ia] = b;
		b->nei[jb] = a;
		//std::cout<<"\n connect "; printTetrahedronVertices(a);
		//std::cout<<"\n     and "; printTetrahedronVertices(b);
		//std::cout<<std::endl;
		return true;
	}
	
	std::cout<<"\n\n [WARNING] failed to connect "; printTetrahedronVertices(a);
	std::cout<<"\n                        and "; printTetrahedronVertices(b);
	std::cout<<std::endl;
		
	return false;
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
	ITetrahedron * nei1;
	ITetrahedron * nei2;
	ITetrahedron * nei3;
	ITetrahedron * nei4;
	ITetrahedron * nei5;
	ITetrahedron * nei6;
	int iv0, iv1, iv2, iv3, iv4;
	
} Bipyramid;

inline void printBipyramidVertices(const Bipyramid * pyra)
{
	std::cout<<"\n bipyramid ("<<pyra->iv0<<", "
		<<pyra->iv1<<", "<<pyra->iv2<<", "<<pyra->iv3<<", "
		<<pyra->iv4<<") ";
}

inline void printBipyramidNeighbors(const Bipyramid * pyra)
{
	if(pyra->nei1) {
	    std::cout<<"\n nei1 "; printTetrahedronVertices(pyra->nei1);
	}
	if(pyra->nei2) {
	    std::cout<<"\n nei2 "; printTetrahedronVertices(pyra->nei2);
	}
	if(pyra->nei3) {
	    std::cout<<"\n nei3 "; printTetrahedronVertices(pyra->nei3);
	}
	if(pyra->nei4) {
	    std::cout<<"\n nei4 "; printTetrahedronVertices(pyra->nei4);
	}
	if(pyra->nei5) {
	    std::cout<<"\n nei5 "; printTetrahedronVertices(pyra->nei5);
	}
	if(pyra->nei6) {
	    std::cout<<"\n nei6 "; printTetrahedronVertices(pyra->nei6);
	}
}

inline void resetBipyramid(Bipyramid * pyra)
{ pyra->tb = NULL; }

inline ITetrahedron * neighborOfBipyramid(const Bipyramid * pyra, int i)
{ 
	if(i==1) return pyra->nei1;
	if(i==2) return pyra->nei2;
	if(i==3) return pyra->nei3;
	if(i==4) return pyra->nei4;
	if(i==5) return pyra->nei5;
	return pyra->nei6;
}

inline bool createBipyramid(Bipyramid * pyra, 
						ITetrahedron * ta, 
						ITetrahedron * tb)
{
	std::cout<<"\n create bipyramid "; printTetrahedronVertices(ta); printTetrahedronVertices(tb);
	pyra->ta = ta;
	pyra->tb = tb;
	int ia, jb;
	if(!findSharedFace(ia, jb, ta, tb) ) 
		return false;
		
	ITRIANGLE tria;
	faceOfTetrahedron(&tria, ta, ia);
	
	pyra->iv0 = oppositeVertex(ta, tria.p1, tria.p2, tria.p3);
	pyra->iv1 = tria.p1;
	pyra->iv2 = tria.p2;
	pyra->iv3 = tria.p3;
	pyra->iv4 = oppositeVertex(tb, tria.p1, tria.p2, tria.p3);
	
	printBipyramidVertices(pyra);
	
	ITRIANGLE tri; int side;
	setTriangleVertices(&tri, pyra->iv0, pyra->iv1, pyra->iv3);
	side = findTetrahedronFace(ta, &tri);
	if(side > -1)
		pyra->nei1 = ta->nei[side];
	else
		printTetrahedronHasNoFace(ta, &tri);
	
	setTriangleVertices(&tri, pyra->iv0, pyra->iv2, pyra->iv1);
	side = findTetrahedronFace(ta, &tri);
	if(side > -1)
		pyra->nei2 = ta->nei[side];
	else
		printTetrahedronHasNoFace(ta, &tri);
				
	setTriangleVertices(&tri, pyra->iv0, pyra->iv3, pyra->iv2);
	side = findTetrahedronFace(ta, &tri);
	if(side > -1)
		pyra->nei3 = ta->nei[side];
	else
		printTetrahedronHasNoFace(ta, &tri);
	
	setTriangleVertices(&tri, pyra->iv4, pyra->iv1, pyra->iv2);
	side = findTetrahedronFace(tb, &tri);
	if(side > -1)
		pyra->nei4 = tb->nei[side];
	else
		printTetrahedronHasNoFace(tb, &tri);

	setTriangleVertices(&tri, pyra->iv4, pyra->iv2, pyra->iv3);
	side = findTetrahedronFace(tb, &tri);
	if(side > -1)
		pyra->nei5 = tb->nei[side];
	else
		printTetrahedronHasNoFace(tb, &tri);
	
	setTriangleVertices(&tri, pyra->iv4, pyra->iv3, pyra->iv1);
	side = findTetrahedronFace(tb, &tri);
	if(side > -1)
		pyra->nei6 = tb->nei[side];
	else
		printTetrahedronHasNoFace(tb, &tri);
		
	return true;
}

/// split t1 to t1 t2 t3 t4 by vi
inline void splitTetrahedron(ITetrahedron * t1, ITetrahedron * t2, 
							ITetrahedron * t3, ITetrahedron * t4,
							int vi)
{
	std::cout<<"\n split "; printTetrahedronVertices(t1);
	
/// vertices of t1		
	const int v0 = t1->iv0;
	const int v1 = t1->iv1;
	const int v2 = t1->iv2;
	const int v3 = t1->iv3;
		
	setTetrahedronVertices(*t1, vi, v1, v2, v3);
	setTetrahedronVertices(*t2, vi, v0, v1, v3);
	setTetrahedronVertices(*t3, vi, v0, v2, v1);
	setTetrahedronVertices(*t4, vi, v0, v3, v2);

	connectTetrahedrons(t1, t2);
	connectTetrahedrons(t1, t3);
	connectTetrahedrons(t1, t4);
	
	connectTetrahedrons(t2, t3);
	connectTetrahedrons(t2, t4);
	
	connectTetrahedrons(t3, t4);
	std::cout<<"\n end split ";
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
	
	ITetrahedron * nei1 = pyra.nei1;		
	ITetrahedron * nei2 = pyra.nei2;
	ITetrahedron * nei3 = pyra.nei3;
	ITetrahedron * nei4 = pyra.nei4;
	ITetrahedron * nei5 = pyra.nei5;
	ITetrahedron * nei6 = pyra.nei6;
	if(!nei1 || !nei2
		|| !nei3 || !nei4
		|| !nei5 || !nei6 )
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
	ITetrahedron * nei1 = pyra.nei1;		
	ITetrahedron * nei2 = pyra.nei2;
	ITetrahedron * nei3 = pyra.nei3;
	ITetrahedron * nei4 = pyra.nei4;
	ITetrahedron * nei5 = pyra.nei5;
	ITetrahedron * nei6 = pyra.nei6;
	if(!nei1 || !nei2
		|| !nei3 || !nei4
		|| !nei5 || !nei6 )
		return false;

	aphid::Vector3F p0 = X[pyra.iv0];
	aphid::Vector3F p1 = X[pyra.iv1];
	aphid::Vector3F p2 = X[pyra.iv2];
	aphid::Vector3F p3 = X[pyra.iv3];
	aphid::Vector3F p4 = X[pyra.iv4];
	TetSphere circ;
	
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
	ITetrahedron * nei1 = pyra.nei1;		
	ITetrahedron * nei2 = pyra.nei2;
	ITetrahedron * nei3 = pyra.nei3;
	ITetrahedron * nei4 = pyra.nei4;
	ITetrahedron * nei5 = pyra.nei5;
	ITetrahedron * nei6 = pyra.nei6;
	
	setTetrahedronVertices(*pyra.ta, v0, v1, v2, v4);
	setTetrahedronVertices(*pyra.tb, v0, v2, v3, v4);

/// add a new one
	ITetrahedron * tc = &tets[numTets++];
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
}

/// flip edge by merge a bipyramid with a neighbor tetrahedron into two tetrahedrons
inline void processMergeFlip(Bipyramid & pyra,
							const int & side,
							ITetrahedron * tets,
							int & numTets)
{
	std::cout<<"\n merge flip"; printBipyramidVertices(&pyra);
	ITetrahedron * tc = neighborOfBipyramid(&pyra, side);
	std::cout<<" and"; printTetrahedronVertices(tc);
	
	ITetrahedron * ta = pyra.ta;
	ITetrahedron * tb = pyra.tb;
	
	const int v0 = pyra.iv0;
	const int v1 = pyra.iv1;
	const int v2 = pyra.iv2;
	const int v3 = pyra.iv3;
	const int v4 = pyra.iv4;
	
	ITetrahedron * nei1 = pyra.nei1;		
	ITetrahedron * nei2 = pyra.nei2;
	ITetrahedron * nei3 = pyra.nei3;
	ITetrahedron * nei4 = pyra.nei4;
	ITetrahedron * nei5 = pyra.nei5;
	ITetrahedron * nei6 = pyra.nei6;

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
	if(tb != &tets[numTets-1]) {
		std::cout<<"\n remove "; printTetrahedronVertices(tb); printTetrahedronVertices(&tets[numTets-1]);
		*tb = tets[numTets-1]; 
	}
/// reduce num tetrahedrons by one
	numTets--;
	
	std::cout<<"\n aft "; printTetrahedronVertices(ta);
	std::cout<<"\n   + "; printTetrahedronVertices(tc);
}

/// once an face is flipped, spawn three more dipyramids for potential face flipping 
inline void spawnFaces(std::deque<Bipyramid> & pyras,
						ITetrahedron * tc)
{
	Bipyramid p0 = pyras[0];
	std::cout<<"\n spawn "; printBipyramidVertices(&p0);
	
	ITetrahedron * nei4 = p0.nei4;
	ITetrahedron * nei5 = p0.nei5;
	ITetrahedron * nei6 = p0.nei6;
	if(nei4) {std::cout<<"\n side4 "; printTetrahedronVertices(p0.ta);
		Bipyramid pa;
		if(createBipyramid(&pa, p0.ta, nei4) )
			pyras.push_back(pa);
	}
	if(nei5) {std::cout<<"\n side5 ";
		Bipyramid pb;
		if(createBipyramid(&pb, p0.tb, nei5) )
			pyras.push_back(pb);
	}
	if(nei6) {std::cout<<"\n side6 ";
		Bipyramid pc;
		if(createBipyramid(&pc, tc, nei6) )
			pyras.push_back(pc);
	}
}

inline void flipFaces(std::deque<Bipyramid> & pyras, 
							const aphid::Vector3F * X,
							ITetrahedron * tets,
							int & numTets)
{
	int nq = pyras.size();
	int mergNei, i=0;
	while(nq>0) {
		if(canSplitFlip(pyras[0], X) ) {
			processSplitFlip(pyras[0], tets, numTets);
			spawnFaces(pyras, &tets[numTets-1] );
		}
		else if(canMergeFlip(mergNei, pyras[0], X) ) {
			processMergeFlip(pyras[0], mergNei, 
							tets, numTets);
		}
		pyras.erase(pyras.begin() );
		nq = pyras.size();
	}
	std::cout<<"\n end flip";
}

}
#endif
