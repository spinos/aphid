/*
 *  diamond_flip.h
 *  foo
 *
 *  Created by jian zhang on 6/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_DIAMOND_FLIP_H
#define TTG_DIAMOND_FLIP_H
#include "triangulation.h"
#include "tetrahedron_graph.h"

namespace ttg {

/// two bipyramids sharing same apex and antiapex
/// can flip four tetra together
struct Diamond {
	ITetrahedron * ta;
	ITetrahedron * tb;
	ITetrahedron * tc;
	ITetrahedron * td;
	int iv0, iv1, iv2, iv3, iv4, iv5;
};

inline void printDiamondVertices(const Diamond * d)
{
	std::cout<<" diamond ("<<d->iv0<<", "<<d->iv1<<", "<<d->iv2<<", "<<d->iv3<<", "
		<<d->iv4<<", "<<d->iv5<<") ";
}

inline void printDiamondTetraVertices(const Diamond * d)
{
	std::cout<<"\n ta "; printTetrahedronVertices(d->ta); 
	std::cout<<"\n tb "; printTetrahedronVertices(d->tb); 
	std::cout<<"\n tc "; printTetrahedronVertices(d->tc); 
	std::cout<<"\n td "; printTetrahedronVertices(d->td); 
}

inline void printDiamondNeighbors(const Diamond * d)
{
	int v0 = d->iv0;
	int v1 = d->iv1;
	int v2 = d->iv2;
	int v3 = d->iv3;
	int v4 = d->iv4;
	int v5 = d->iv5;
	ITetrahedron * nei1 = neighborOfTetrahedron(d->ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(d->ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(d->tb, v4, v1, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(d->tb, v4, v3, v1);
	ITetrahedron * nei5 = neighborOfTetrahedron(d->tc, v0, v3, v5);
	ITetrahedron * nei6 = neighborOfTetrahedron(d->tc, v0, v5, v2);
	ITetrahedron * nei7 = neighborOfTetrahedron(d->td, v4, v2, v5);
	ITetrahedron * nei8 = neighborOfTetrahedron(d->td, v4, v5, v3);
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
	if(nei7) {
		std::cout<<"\n nei7 "; printTetrahedronVertices(nei7); 
	}
	if(nei8) {
		std::cout<<"\n nei8 "; printTetrahedronVertices(nei8); 
	}
}

inline void resetDiamond(Diamond * d)
{ d->ta = NULL; d->iv5 = -1; }

inline bool isDiamondComplete(const Diamond * d)
{ return d->iv5 > -1; }

inline bool matchDiamond(Diamond * diam, 
						const Bipyramid * pyra)
{
	if(pyra->iv0 != diam->iv0
		&& pyra->iv0 != diam->iv4)
			return false;
			
	if(pyra->iv4 != diam->iv0
		&& pyra->iv4 != diam->iv4)
			return false;
			
	ITRIANGLE tria, trib;
	setTriangleVertices(&tria, diam->iv1, diam->iv2, diam->iv3 );
	setTriangleVertices(&trib, pyra->iv1, pyra->iv2, pyra->iv3 );
	
	int ea, eb;
	if(!trianglesShareEdge(ea, eb, &tria, &trib) ) 
		return false;
		
	diam->iv5 = oppositeVertex(&trib, ea, eb);
	diam->iv1 = oppositeVertex(&tria, ea, eb);
	IEDGE e = findOppositeEdge(&tria, diam->iv1);
	diam->iv2 = e.v[0];
	diam->iv3 = e.v[1];
	return true;
}

inline void addToDiamond(Diamond * diam, 
						const Bipyramid * pyra)
{
	if(!diam->ta) {
		diam->ta = pyra->ta;
		diam->tb = pyra->tb;
		diam->iv0 = pyra->iv0;
		diam->iv1 = pyra->iv1;
		diam->iv2 = pyra->iv2;
		diam->iv3 = pyra->iv3;
		diam->iv4 = pyra->iv4;
	}
	else {
		if(!matchDiamond(diam, pyra) ) 
			return;
			
		if(tetrahedronHasVertex(pyra->ta, diam->iv0) > -1) {
			diam->tc = pyra->ta;
			diam->td = pyra->tb;
		}
		else {
			diam->td = pyra->ta;
			diam->tc = pyra->tb;
		}
	}
}

///   b4        aft
/// a 0 1 2 3   2 0 1 4
/// b 4 1 3 2   3 0 4 1
/// c 0 2 5 3   2 0 4 5    
/// d 4 2 3 5   3 0 5 4
///   face      b4  aft
/// 1 0 1 3     a   b
/// 2 0 2 1     a   a
/// 3 4 1 2     b   a 
/// 4 4 3 1     b   b
/// 5 0 3 5     c   d
/// 6 0 5 2     c   c
/// 7 4 2 5     d   c
/// 8 4 5 3     d   d
inline void flipDiamond(Diamond * diam)
{
	std::cout<<"\n [INFO] 4way flip "; printDiamondVertices(diam);
	
	int v0 = diam->iv0;
	int v1 = diam->iv1;
	int v2 = diam->iv2;
	int v3 = diam->iv3;
	int v4 = diam->iv4;
	int v5 = diam->iv5;
	ITetrahedron * nei1 = neighborOfTetrahedron(diam->ta, v0, v1, v3);
	ITetrahedron * nei2 = neighborOfTetrahedron(diam->ta, v0, v2, v1);
	ITetrahedron * nei3 = neighborOfTetrahedron(diam->tb, v4, v1, v2);
	ITetrahedron * nei4 = neighborOfTetrahedron(diam->tb, v4, v3, v1);
	ITetrahedron * nei5 = neighborOfTetrahedron(diam->tc, v0, v3, v5);
	ITetrahedron * nei6 = neighborOfTetrahedron(diam->tc, v0, v5, v2);
	ITetrahedron * nei7 = neighborOfTetrahedron(diam->td, v4, v2, v5);
	ITetrahedron * nei8 = neighborOfTetrahedron(diam->td, v4, v5, v3);
	
	setTetrahedronVertices(*diam->ta, v2, v0, v1, v4);
	setTetrahedronVertices(*diam->tb, v3, v0, v4, v1);
	setTetrahedronVertices(*diam->tc, v2, v0, v4, v5);
	setTetrahedronVertices(*diam->td, v3, v0, v5, v4);
	
	resetTetrahedronNeighbors(*diam->ta);
	resetTetrahedronNeighbors(*diam->tb);
	resetTetrahedronNeighbors(*diam->tc);
	resetTetrahedronNeighbors(*diam->td);
	
	connectTetrahedrons(diam->ta, diam->tb);
	connectTetrahedrons(diam->ta, diam->tc);
	connectTetrahedrons(diam->tb, diam->td);
	connectTetrahedrons(diam->tc, diam->td);
	
	if(nei1)
		connectTetrahedrons(nei1, diam->tb);
	if(nei2)
		connectTetrahedrons(nei2, diam->ta);
	if(nei3)
		connectTetrahedrons(nei3, diam->ta);
	if(nei4)
		connectTetrahedrons(nei4, diam->tb);
	if(nei5)
		connectTetrahedrons(nei5, diam->td);
	if(nei6)
		connectTetrahedrons(nei6, diam->tc);
	if(nei7)
		connectTetrahedrons(nei7, diam->tc);
	if(nei8)
		connectTetrahedrons(nei8, diam->td);
		
	std::cout<<"\n edge ("<<v0<<", "<<v4<<") ";
}

}

#endif