/*
 *  tetrahedron_graph.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_TETRAHEDRON_GRAPH_H
#define TTG_TETRAHEDRON_GRAPH_H

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

inline bool tetrahedronHas6Neighbors(const ITetrahedron * t)
{
    if(!t->nei0) return false; 
    if(!t->nei1) return false; 
    if(!t->nei2) return false;
    if(!t->nei3) return false; 
	if(t->nei0->index < 0) return false;
	if(t->nei1->index < 0) return false;
	if(t->nei2->index < 0) return false;
	if(t->nei3->index < 0) return false;
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

inline int tetrahedronHasVertex(const ITetrahedron * t,
								const int & x)
{
	if(t->iv0 == x) return 0;
	if(t->iv1 == x) return 1;
	if(t->iv2 == x) return 2;
	if(t->iv3 == x) return 3;
	return -1;
}

inline int oppositeVertex(const ITetrahedron * t, 
						int a, int b, int c)
{
	if(t->iv0!=a && t->iv0 != b && t->iv0 != c) return t->iv0;
	if(t->iv1!=a && t->iv1 != b && t->iv1 != c) return t->iv1;
	if(t->iv2!=a && t->iv2 != b && t->iv2 != c) return t->iv2;
	
	return t->iv3;
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

inline bool connectTetrahedrons(ITetrahedron * a, ITetrahedron * b,
						int v1, int v2, int v3)
{
	ITRIANGLE trif;
	setTriangleVertices(&trif, v1, v2, v3);
	int ia = findTetrahedronFace(a, &trif);
	if(ia < 0) {
		return false;
	}
	
	int jb = findTetrahedronFace(b, &trif);
	if(jb < 0) {
		return false;
	}
	
	setTetrahedronNeighbor(a, b, ia);
	setTetrahedronNeighbor(b, a, jb);
	
	return true;
}

inline bool connectTetrahedrons(ITetrahedron * a, ITetrahedron * b)
{
	if(!a) return false;
	if(!b) return false;
	if(a->index<0) return false;
	if(b->index<0) return false;
	
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

inline bool checkTetrahedronConnections(ITetrahedron * a)
{
	if(!a) return false;
	if(a->index < 0) return false;
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

inline void printTetrahedronHasNoFace(const ITetrahedron * a, const ITRIANGLE * f)
{
	std::cout<<"\n\n [WARNING] no face "; printTriangleVertice(f);
	std::cout<<"\n				in "; printTetrahedronVertices(a);
}

inline void printTetrahedronCannotConnect(const ITetrahedron * a, 
						const ITetrahedron * b)
{
	std::cout<<"\n\n [ERROR] cannot connect "; printTetrahedronVertices(a);
	std::cout<<"\n                      and "; printTetrahedronVertices(b);
}

}

#endif