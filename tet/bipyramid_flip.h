/*
 *  bipyramid_flip.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_BIPYRAMID_FLIP_H
#define TTG_BIPYRAMID_FLIP_H
#include <triangle_math.h>
#include <tetrahedron_math.h>
#include "triangulation.h"
#include "tetrahedron_graph.h"

namespace ttg {
								
typedef struct {
   aphid::Vector3F pc;
   float r;
} TetSphere;
								
/// two or three tetrahedrons sharing same face (1, 2, 3)
/// and six neighbors
typedef struct {
	ITetrahedron * ta;
	ITetrahedron * tb;
	ITetrahedron * tc;
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

inline void sideOfBipyramidConnection(int & a, int & b, int & c,
							const Bipyramid & pyra,
							int side)
{
	if(side == 1) {
		a = pyra.iv0; b = pyra.iv1; c = pyra.iv3;
	}
	else if(side == 2) {
		a = pyra.iv0; b = pyra.iv2; c = pyra.iv1;
	}
	else if(side == 3) {
		a = pyra.iv0; b = pyra.iv2; c = pyra.iv3;
	}
	else if(side == 4) {
		a = pyra.iv4; b = pyra.iv2; c = pyra.iv1;
	}
	else if(side == 5) {
		a = pyra.iv4; b = pyra.iv3; c = pyra.iv2;
	}
	else {
		a = pyra.iv4; b = pyra.iv3; c = pyra.iv1;
	}
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
	
	if(!nei1) return false;
	if(!nei2) return false;
	if(!nei3) return false;
	if(!nei4) return false;
	if(!nei5) return false;
	if(!nei6) return false;
	
	if(tetrahedronHasVertex(nei1, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei2, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei3, pyra.iv4) > -1 ) return false;
	if(tetrahedronHasVertex(nei4, pyra.iv0) > -1 ) return false;
	if(tetrahedronHasVertex(nei5, pyra.iv0) > -1 ) return false;
	if(tetrahedronHasVertex(nei6, pyra.iv0) > -1 ) return false;
	
	return true;
}

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
	if(ta->index < 0 || tb->index < 0) {
		std::cout<<"\n [WARNING] not connected ";
		printTetrahedronVertices(ta); printTetrahedronVertices(tb);
		return false;
	}
	pyra->ta = ta;
	pyra->tb = tb;
	int ia, jb;
	if(!findSharedFace(ia, jb, ta, tb) ) {
		std::cout<<"\n [WARNING] not connected ";
		printTetrahedronVertices(ta); printTetrahedronVertices(tb);
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

	if(!checkTetrahedronConnections(ta) ) {
		if(ta) printTetrahedronNeighbors(ta);
		return false;
	}
	if(!checkTetrahedronConnections(tb) ) {
		if(tb) printTetrahedronNeighbors(tb);
		return false;
	}

	std::cout<<"\n success "<<std::endl;
	return true;
}

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

/// flip edge by split bipyramid into three tetrahedrons
inline bool processSplitFlip(Bipyramid & pyra,
							ITetrahedron * tets,
							int & numTets)
{
	std::cout<<"\n split flip"; printBipyramidVertices(&pyra);
	
	//printTetrahedronVertices(pyra.ta);
	//printTetrahedronVertices(pyra.tb);
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
		return false;
	}
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tb) ) printTetrahedronNeighbors(pyra.tb);
	
	setTetrahedronVertices(*pyra.ta, v0, v1, v2, v4);
	setTetrahedronVertices(*pyra.tb, v0, v2, v3, v4);
	
	
/// add a new one
	pyra.tc = &tets[numTets]; pyra.tc->index = numTets;
	numTets++;
	setTetrahedronVertices(*pyra.tc, v0, v1, v4, v3);
	
	connectTetrahedrons(pyra.ta, pyra.tb);
	connectTetrahedrons(pyra.ta, pyra.tc);
	connectTetrahedrons(pyra.tc, pyra.tb);

	connectTetrahedrons(nei2, pyra.ta);
	connectTetrahedrons(nei4, pyra.ta);
	connectTetrahedrons(nei3, pyra.tb);
	connectTetrahedrons(nei5, pyra.tb);
	connectTetrahedrons(nei1, pyra.tc);
	connectTetrahedrons(nei6, pyra.tc);
	
	std::cout<<"\n edge ("<<v0<<", "<<v4<<")";
	std::cout<<"\n aft "; printTetrahedronVertices(pyra.ta);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tb);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tc);
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tb) ) printTetrahedronNeighbors(pyra.tb);
	if(!checkTetrahedronConnections(pyra.tc) ) printTetrahedronNeighbors(pyra.tc);
	return true;
}

/// flip edge by merge a bipyramid with a neighbor tetrahedron into two tetrahedrons
inline bool processMergeFlip(Bipyramid * pyra,
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

	//ITetrahedron * tc;
	if(side == 1) pyra->tc = nei1;
	else if(side == 2) pyra->tc = nei2;
	else pyra->tc = nei3;
	std::cout<<" and"; printTetrahedronVertices(pyra->tc);
	
/// connection to c	
	ITetrahedron * neica;
	ITetrahedron * neicc;
	
	if(side == 1) {
		neica = neighborOfTetrahedron(pyra->tc, v0, v1, v4);
		neicc = neighborOfTetrahedron(pyra->tc, v0, v4, v3);
		
		setTetrahedronVertices(*ta, v1, v2, v0, v4);
		setTetrahedronVertices(*pyra->tc, v3, v4, v0, v2);

		connectTetrahedrons(ta, nei2);
		connectTetrahedrons(ta, nei4);
		connectTetrahedrons(pyra->tc, nei3);
		connectTetrahedrons(pyra->tc, nei5);
	}
	else if(side == 2) {
		neica = neighborOfTetrahedron(pyra->tc, v0, v2, v4);
		neicc = neighborOfTetrahedron(pyra->tc, v0, v4, v1);
		
		setTetrahedronVertices(*ta, v2, v3, v0, v4);
		setTetrahedronVertices(*pyra->tc, v1, v4, v0, v3);
		
		connectTetrahedrons(ta, nei3);
		connectTetrahedrons(ta, nei5);
		connectTetrahedrons(pyra->tc, nei1);
		connectTetrahedrons(pyra->tc, nei6);
	}
	else {
		neica = neighborOfTetrahedron(pyra->tc, v0, v3, v4);
		neicc = neighborOfTetrahedron(pyra->tc, v0, v4, v2);
		
		setTetrahedronVertices(*ta, v3, v1, v0, v4);
		setTetrahedronVertices(*pyra->tc, v2, v4, v0, v1);
		
		connectTetrahedrons(ta, nei1);
		connectTetrahedrons(ta, nei6);
		connectTetrahedrons(pyra->tc, nei2);
		connectTetrahedrons(pyra->tc, nei4);
	}

	connectTetrahedrons(ta, pyra->tc);
	connectTetrahedrons(ta, neica);
	connectTetrahedrons(pyra->tc, neicc);
	
/// remove tb
	// std::cout<<"\n remove "; printTetrahedronVertices(tb); 
	pyra->tb->index = -1;
	
	std::cout<<"\n aft "; printTetrahedronVertices(ta);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra->tc);
	
	if(!checkTetrahedronConnections(pyra->tc)) {
		std::cout<<"\n [ERROR] wrong tetrahedron connections"; printTetrahedronVertices(pyra->tc);
		return false;
	}
	
	if(!checkTetrahedronConnections(pyra->ta) ) printTetrahedronNeighbors(pyra->ta);
	if(!checkTetrahedronConnections(pyra->tc) ) printTetrahedronNeighbors(pyra->tc);
	
	createBipyramid(pyra, pyra->ta, pyra->tc);
	
	return true;
}

inline void spawnOnSide(std::deque<Bipyramid> & pyras,
						const Bipyramid & p0,
						ITetrahedron * tup,
						int side)
{
	std::cout<<"\n spawn "; printBipyramidVertices(&p0);
	std::cout<<"\n side "<<side;
	
	int v1, v2, v3;
	sideOfBipyramidConnection(v1, v2, v3, p0, side);
							
	ITetrahedron * nei = neighborOfTetrahedron(tup, v1, v2, v3);
	
	if(nei) {
		Bipyramid pa;
		if(createBipyramid(&pa, tup, nei) )
			pyras.push_back(pa);
	}
}

inline void addFace(std::vector<IFace *> & faces,
					ITetrahedron * ta,
					ITetrahedron * tb,
					const int & va, const int & vb, const int & vc)
{
	IFace * tri = new IFace;
	tri->key = aphid::sdb::Coord3(va, vb, vc);
	tri->ta = ta;
	tri->tb = tb;
	faces.push_back(tri);
}

inline bool canFaceFlip(IFace * f)
{
	if(f->ta->index < 0) return false;
	if(f->tb->index < 0) return false;
	if(!canConnectTetrahedrons(f->ta, f->tb,
						f->key.x, f->key.y, f->key.z) ) return false;
	
	return true;
}

inline bool createBipyramid1(Bipyramid & pyra, 
						ITetrahedron * ta, 
						ITetrahedron * tb)
{
	pyra.ta = ta;
	pyra.tb = tb;
	int ia, jb;
	findSharedFace(ia, jb, ta, tb);		
	ITRIANGLE tria;
	faceOfTetrahedron(&tria, ta, ia);
	
	pyra.iv0 = oppositeVertex(ta, tria.p1, tria.p2, tria.p3);
	pyra.iv1 = tria.p1;
	pyra.iv2 = tria.p2;
	pyra.iv3 = tria.p3;
	pyra.iv4 = oppositeVertex(tb, tria.p1, tria.p2, tria.p3);
	
	// std::cout<<"\n create "; printBipyramidVertices(&pyra);

	return true;
}

inline bool canSplitFlip1(const Bipyramid & pyra, 
							const aphid::Vector3F * X)
{		
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
	if(antipex.distanceTo(circ.pc) >= circ.r)
		return false;

	return true;
}

/// check low volume
inline bool checkCoplanar(const Bipyramid & pyra, 
							const aphid::Vector3F * X)
{
	aphid::Vector3F a = X[pyra.iv1];
	aphid::Vector3F b = X[pyra.iv2];
	aphid::Vector3F c = X[pyra.iv3];
	aphid::Vector3F apex = X[pyra.iv0];
	aphid::Vector3F antipex = X[pyra.iv4];
	
	const float lowVol = (tetrahedronVolume1(apex, a, b, c)
						+ tetrahedronVolume1(antipex, c, b, a) ) * .03f;
	
	float vol = tetrahedronVolume1(apex, a, b, antipex);
	if(vol < lowVol) {
		std::cout<<"\n low volume "<<vol;
		printBipyramidVertices(&pyra);
		return true;
	}
	
	vol	= tetrahedronVolume1(apex, b, c, antipex);
	if(vol < lowVol) {
		std::cout<<"\n low volume "<<vol;
		printBipyramidVertices(&pyra);
		return true;
	}

	vol = tetrahedronVolume1(apex, a, antipex, c);
	if(vol < lowVol) {
		std::cout<<"\n low volume "<<vol;
		printBipyramidVertices(&pyra);
		return true;
	}
	
	return false;
}

/// flip edge by split bipyramid into three tetrahedrons
inline bool processSplitFlip1(Bipyramid & pyra,
							std::vector<ITetrahedron *> & tets,
							std::vector<IFace *> & boundary)
{
	std::cout<<"\n split flip"; printBipyramidVertices(&pyra);
	
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
/*
	printTetrahedronVertices(pyra.ta);
	printTetrahedronVertices(pyra.tb);
	
	if(nei1) {
		std::cout<<"\n nei1 "; printTetrahedronVertices(nei1);
	}
	if(nei2) {
		std::cout<<"\n nei2 "; printTetrahedronVertices(nei2);
	}
	if(nei3) {
		std::cout<<"\n nei3 "; printTetrahedronVertices(nei3);
	}
	if(nei4) {
		std::cout<<"\n nei4 "; printTetrahedronVertices(nei4);
	}
	if(nei5) {
		std::cout<<"\n nei5 "; printTetrahedronVertices(nei5);
	}
	if(nei6) {
		std::cout<<"\n nei6 "; printTetrahedronVertices(nei6);
	}
*/
	setTetrahedronVertices(*pyra.ta, v0, v1, v2, v4);
	setTetrahedronVertices(*pyra.tb, v0, v2, v3, v4);
	
/// add a new one
	ITetrahedron * tc = new ITetrahedron;
	setTetrahedronVertices(*tc, v0, v1, v4, v3);
	tc->index = tets.size();
	tets.push_back(tc);
	pyra.tc = tc;
	
	resetTetrahedronNeighbors(*pyra.ta);
	resetTetrahedronNeighbors(*pyra.tb);
	resetTetrahedronNeighbors(*tc);

	connectTetrahedrons(pyra.ta, pyra.tb);
	connectTetrahedrons(pyra.ta, pyra.tc);
	connectTetrahedrons(pyra.tc, pyra.tb);

/// spawn 6 boundary faces
	if(nei2) {
		//std::cout<<"\n connect n2 a"; 
		connectTetrahedrons(nei2, pyra.ta);
		addFace(boundary, pyra.ta, nei2, v0, v2, v1);
	}
	if(nei4) {
		//std::cout<<"\n connect n4 a";
		connectTetrahedrons(nei4, pyra.ta);
		addFace(boundary, pyra.ta, nei4, v1, v2, v4);
	}
	if(nei3) {
		//std::cout<<"\n connect n3 b";
		connectTetrahedrons(nei3, pyra.tb);
		addFace(boundary, pyra.tb, nei3, v0, v3, v2);
	}
	if(nei5) {
		//std::cout<<"\n connect n5 b";
		connectTetrahedrons(nei5, pyra.tb);
		addFace(boundary, pyra.tb, nei5, v3, v4, v2);
	}
	if(nei1) {
		//std::cout<<"\n connect n1 c";
		connectTetrahedrons(nei1, pyra.tc);
		addFace(boundary, pyra.tc, nei1, v0, v1, v3);
	}
	if(nei6) {
		//std::cout<<"\n connect n6 c";
		connectTetrahedrons(nei6, pyra.tc);
		addFace(boundary, pyra.tc, nei6, v1, v4, v3);
	}
	
	std::cout<<"\n edge ("<<v0<<", "<<v4<<")";
/*
	std::cout<<"\n aft "; printTetrahedronVertices(pyra.ta);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tb);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tc);
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tb) ) printTetrahedronNeighbors(pyra.tb);
	if(!checkTetrahedronConnections(pyra.tc) ) printTetrahedronNeighbors(pyra.tc);
*/	
	return true;
}

inline bool canMergeFlip1(int & i3rd, const Bipyramid & pyra, 
							const aphid::Vector3F * X)
{
	const int v0 = pyra.iv0;
	const int v1 = pyra.iv1;
	const int v2 = pyra.iv2;
	const int v3 = pyra.iv3;
	const int v4 = pyra.iv4;
	
	aphid::Vector3F p0 = X[v0];
	aphid::Vector3F p1 = X[v1];
	aphid::Vector3F p2 = X[v2];
	aphid::Vector3F p3 = X[v3];
	aphid::Vector3F p4 = X[v4];
	TetSphere circ;
	
	ITetrahedron * nei1 = neighborOfTetrahedron(pyra.ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(pyra.ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(pyra.ta, v0, v3, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(pyra.tb, v4, v1, v2);
	ITetrahedron * nei5 = neighborOfTetrahedron(pyra.tb, v4, v2, v3);
	ITetrahedron * nei6 = neighborOfTetrahedron(pyra.tb, v4, v3, v1);

/*	
	if(nei1) {
		std::cout<<"\n nei1 "; printTetrahedronVertices(nei1);
	}
	if(nei2) {
		std::cout<<"\n nei2 "; printTetrahedronVertices(nei2);
	}
	if(nei3) {
		std::cout<<"\n nei3 "; printTetrahedronVertices(nei3);
	}
	if(nei4) {
		std::cout<<"\n nei4 "; printTetrahedronVertices(nei4);
	}
	if(nei5) {
		std::cout<<"\n nei5 "; printTetrahedronVertices(nei5);
	}
	if(nei6) {
		std::cout<<"\n nei6 "; printTetrahedronVertices(nei6);
	}
*/
	
/// find shared neighbor pair	
	if(nei1) {
		if(tetrahedronHasVertex(nei1, v4) > -1 ) {
			
/// be convex
		if(!aphid::segmentIntersectTriangle(p3, p1, p2, p4, p0) )
			return false;

/// empty principle
#define CHECK_EMPTY_PRINC 1
#if CHECK_EMPTY_PRINC
		circumSphere(circ, p3, p2, p4, p0 );
		if(p1.distanceTo(circ.pc) < circ.r )
			return false;
#endif
			
		i3rd = 1;
		return true;
		
		}
	}
	
	if(nei2) { 
		if(tetrahedronHasVertex(nei2, v4) > -1 ) {
			
		if(!aphid::segmentIntersectTriangle(p1, p2, p3, p4, p0) )
			return false;

#if CHECK_EMPTY_PRINC		
		circumSphere(circ, p0, p1, p4, p3 );
		if(p2.distanceTo(circ.pc) < circ.r )
			return false;
#endif
		
		i3rd = 2;
		return true;
		
		}
	}
	
	if(nei3) { 
		if(tetrahedronHasVertex(nei3, v4) > -1 ) {
			
		if(!aphid::segmentIntersectTriangle(p2, p3, p1, p4, p0) )
			return false;

#if CHECK_EMPTY_PRINC		
		circumSphere(circ, p2, p1, p4, p0 );
		if(p3.distanceTo(circ.pc) < circ.r )
			return false;
#endif
		
		i3rd = 3;
		return true;
		
		}
	}
	
	return false;
}

inline bool processMergeFlip1(Bipyramid & pyra,
							std::vector<ITetrahedron *> & tets,
							const int & side,
							std::vector<IFace *> & boundary)
{
	std::cout<<"\n merge flip"; printBipyramidVertices(&pyra);
	
	ITetrahedron * ta = pyra.ta;
	ITetrahedron * tb = pyra.tb;
	
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

	if(side == 1) pyra.tc = nei1;
	else if(side == 2) pyra.tc = nei2;
	else pyra.tc = nei3;
	std::cout<<" and"; printTetrahedronVertices(pyra.tc);
	
/// connection to c	
	ITetrahedron * neica;
	ITetrahedron * neicc;
	
	if(side == 1) {
		neica = neighborOfTetrahedron(pyra.tc, v0, v1, v4);
		neicc = neighborOfTetrahedron(pyra.tc, v0, v4, v3);
		
		setTetrahedronVertices(*ta, v1, v2, v0, v4);
		setTetrahedronVertices(*pyra.tc, v3, v4, v0, v2);
		
		resetTetrahedronNeighbors(*ta);
		resetTetrahedronNeighbors(*pyra.tc);

		connectTetrahedrons(ta, nei2);
		connectTetrahedrons(ta, nei4);
		connectTetrahedrons(pyra.tc, nei3);
		connectTetrahedrons(pyra.tc, nei5);
	}
	else if(side == 2) {
		neica = neighborOfTetrahedron(pyra.tc, v0, v2, v4);
		neicc = neighborOfTetrahedron(pyra.tc, v0, v4, v1);
		
		setTetrahedronVertices(*ta, v2, v3, v0, v4);
		setTetrahedronVertices(*pyra.tc, v1, v4, v0, v3);
		
		resetTetrahedronNeighbors(*ta);
		resetTetrahedronNeighbors(*pyra.tc);
		
		connectTetrahedrons(ta, nei3);
		connectTetrahedrons(ta, nei5);
		connectTetrahedrons(pyra.tc, nei1);
		connectTetrahedrons(pyra.tc, nei6);
	}
	else {
		neica = neighborOfTetrahedron(pyra.tc, v0, v3, v4);
		neicc = neighborOfTetrahedron(pyra.tc, v0, v4, v2);
		
		setTetrahedronVertices(*ta, v3, v1, v0, v4);
		setTetrahedronVertices(*pyra.tc, v2, v4, v0, v1);
		
		resetTetrahedronNeighbors(*ta);
		resetTetrahedronNeighbors(*pyra.tc);
		
		connectTetrahedrons(ta, nei1);
		connectTetrahedrons(ta, nei6);
		connectTetrahedrons(pyra.tc, nei2);
		connectTetrahedrons(pyra.tc, nei4);
	}

	connectTetrahedrons(ta, pyra.tc);
	connectTetrahedrons(ta, neica);
	connectTetrahedrons(pyra.tc, neicc);
	
/// remove tb
	// std::cout<<"\n remove "; printTetrahedronVertices(tb); 
	pyra.tb->index = -1;
	
	std::cout<<"\n aft "; printTetrahedronVertices(ta);
	std::cout<<"\n   + "; printTetrahedronVertices(pyra.tc);
	
	if(!checkTetrahedronConnections(pyra.tc)) {
		std::cout<<"\n [ERROR] wrong tetrahedron connections"; printTetrahedronVertices(pyra.tc);
		return false;
	}
	
	if(!checkTetrahedronConnections(pyra.ta) ) printTetrahedronNeighbors(pyra.ta);
	if(!checkTetrahedronConnections(pyra.tc) ) printTetrahedronNeighbors(pyra.tc);
	
	createBipyramid1(pyra, pyra.ta, pyra.tc);


/// spawn six boundary faces
	v0 = pyra.iv0;
	v1 = pyra.iv1;
	v2 = pyra.iv2;
	v3 = pyra.iv3;
	v4 = pyra.iv4;
	
	nei1 = neighborOfTetrahedron(pyra.ta, v0, v1, v3);
	nei2 = neighborOfTetrahedron(pyra.ta, v0, v2, v1);
	nei3 = neighborOfTetrahedron(pyra.ta, v0, v3, v2);
	nei4 = neighborOfTetrahedron(pyra.tb, v4, v1, v2);
	nei5 = neighborOfTetrahedron(pyra.tb, v4, v2, v3);
	nei6 = neighborOfTetrahedron(pyra.tb, v4, v3, v1);
	
	if(nei1)
		addFace(boundary, pyra.ta, nei1, v0, v1, v3);
	if(nei2)
		addFace(boundary, pyra.ta, nei2, v0, v2, v1);
	if(nei3)
		addFace(boundary, pyra.ta, nei3, v0, v3, v2);
	if(nei4)
		addFace(boundary, pyra.tb, nei4, v4, v1, v2);
	if(nei5)
		addFace(boundary, pyra.tb, nei5, v4, v2, v3);
	if(nei6)
		addFace(boundary, pyra.tb, nei6, v4, v3, v1);
		
	return true;
}

inline void flipAFace(IFace * f,
						std::vector<IFace *> & boundary,
						std::vector<ITetrahedron *> & tets,
						const aphid::Vector3F * X)
{
	std::cout<<"\n flip ("
			<<f->key.x<<", "<<f->key.y<<", "<<f->key.z<<") ";
	
	if(!canFaceFlip(f) )
		return;
	
	Bipyramid pyra;
	if(!createBipyramid1(pyra, f->ta, f->tb) )
		return;
	
	int imerg;
	if(canSplitFlip1(pyra, X) ) {
		if(checkCoplanar(pyra, X) )
		{}
		else
			processSplitFlip1(pyra, tets, boundary);
	}
	else if(canMergeFlip1(imerg, pyra, X) ) {
		processMergeFlip1(pyra, tets, imerg, boundary);
	}
}

inline void flipFaces(std::vector<IFace *> & faces,
						std::vector<ITetrahedron *> & tets,
						const aphid::Vector3F * X)
{
	while(faces.size() >0) {
		
		flipAFace(faces[0], faces, tets, X);
		
		delete faces[0];
		faces.erase(faces.begin() );
	}
}

}
#endif