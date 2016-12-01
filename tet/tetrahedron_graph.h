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

#include <sdb/Array.h>

namespace ttg {

struct ITetrahedron {
	
	ITetrahedron()
	{
		nei0 = nei1 = nei2 = nei3 = NULL;
	}
	
	ITetrahedron * nei0;
	ITetrahedron * nei1;
	ITetrahedron * nei2;
	ITetrahedron * nei3;
	int iv0, iv1, iv2, iv3;
	int index;
};

struct IEdge {

    int iv0, iv1;
    ITetrahedron * nei0;
	ITetrahedron * nei1;
};

struct IFace {
	
	IFace() 
	{
		ta = NULL;
		tb = NULL;
	}
	
	aphid::sdb::Coord3 key;
	ITetrahedron * ta;
	ITetrahedron * tb;
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

/// f0 1 2 3
/// f1 0 1 3
/// f2 0 2 1
/// f3 0 3 2
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

inline bool sharedVertexTetrahedron(const ITetrahedron * a,
									const ITetrahedron * b)
{
	if(tetrahedronHasVertex(a, b->iv0) ) 
		return true;
		
	if(tetrahedronHasVertex(a, b->iv1) ) 
		return true;
		
	if(tetrahedronHasVertex(a, b->iv2) ) 
		return true;
		
	if(tetrahedronHasVertex(a, b->iv3) ) 
		return true;
		
	return false;
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

inline bool canConnectTetrahedrons(const ITetrahedron * a, 
						const ITetrahedron * b,
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
	
	return true;
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
	
	int i, ia, jb;
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

inline bool checkTetrahedronConnections(std::vector<ITetrahedron *> & tets)
{
	std::vector<ITetrahedron *>::iterator it = tets.begin();
	for(;it!=tets.end();++it) {
		if((*it)->index < 0) continue;
		if(!checkTetrahedronConnections(*it) ) {
			std::cout<<"\n [WARNING] failed connectivity check";
			printTetrahedronVertices(*it);
			return false;
		}
	}
	return true;
}

inline ITetrahedron * tetrahedronNeighbor(const ITetrahedron * a, const int & i)
{
    if(i==0) return a->nei0;
    if(i==1) return a->nei1;
    if(i==2) return a->nei2;
    return a->nei3;
}

inline int tetrahedronVertex(const ITetrahedron * a, const int & i)
{
	if(i==0) return a->iv0;
    if(i==1) return a->iv1;
    if(i==2) return a->iv2;
    return a->iv3;
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

inline bool findEdgeNeighborPair(IEdge * e,
                                 const ITetrahedron * a)
{
    bool hasOne = false;
    bool hasTwo = false;
    ITRIANGLE trif;
	int i;
	for(i=0;i<4;++i) {
		faceOfTetrahedron(&trif, a, i);
		if(triangleHasEdge(&trif, e->iv0, e->iv1) ) {
            if(!hasOne) {
                e->nei0 = tetrahedronNeighbor(a, i);
                hasOne = true;
            }
            else {
                e->nei1 = tetrahedronNeighbor(a, i);
                hasTwo = true;
            }
        }
	}
    
    return (hasOne && hasTwo);
}

inline bool findTetrahedronEdge(IEdge * e,
                                const ITetrahedron * a,
                                 const aphid::Float4 & coord)
{
    bool hasOne = false;
    bool hasTwo = false;
    if(coord.x > .03f) {
        e->iv0 = a->iv0;
        hasOne = true;
    }
    
    if(coord.y > .03f) {
        if(!hasOne) {
            e->iv0 = a->iv1;
            hasOne = true;
        }
        else {
            e->iv1 = a->iv1;
            hasTwo = true;
        }
    }
    
    if(coord.z > .03f) {
        if(!hasOne) {
            e->iv0 = a->iv2;
            hasOne = true;
        }
        else {
            e->iv1 = a->iv2;
            hasTwo = true;
        }
    }
    
    if(coord.w > .03f) {
        if(!hasOne) {
            e->iv0 = a->iv3;
            hasOne = true;
        }
        else {
            e->iv1 = a->iv3;
            hasTwo = true;
        }
    }
    
    if(!hasOne || !hasTwo) return false;
    
    return findEdgeNeighborPair(e, a);
}

inline IEdge oppositeEdge(const ITetrahedron * t,
							const IEdge * e)
{
	IEdge ea;
	bool hasOne = false;
    
	if(t->iv0 != e->iv0
		&& t->iv0 != e->iv1) {
		
		ea.iv0 = t->iv0;
		hasOne = true;
	}
	
	if(t->iv1 != e->iv0
		&& t->iv1 != e->iv1) {
		
		if(!hasOne) {
			ea.iv0 = t->iv1;
			hasOne = true;
		}
		else {
			ea.iv1 = t->iv1;
		}
	}
	
	if(t->iv2 != e->iv0
		&& t->iv2 != e->iv1) {
		
		if(!hasOne) {
			ea.iv0 = t->iv2;
			hasOne = true;
		}
		else {
			ea.iv1 = t->iv2;
		}
	}
	
	if(t->iv3 != e->iv0
		&& t->iv3 != e->iv1) {
		
		if(!hasOne) {
			ea.iv0 = t->iv3;
			hasOne = true;
		}
		else {
			ea.iv1 = t->iv3;
		}
	}
	
    return ea;
}

inline ITRIANGLE oppositeFace(const ITetrahedron * t,
								const int & a)
{
	ITRIANGLE trif;
	int i;
	for(i=0;i<4;++i) {
		faceOfTetrahedron(&trif, t, i);
		if(containtsVertex(&trif, a) < 0) 
			return trif;
	}
	
	return trif;
}

inline bool findTetrahedronFace(IFace * f,
								 ITetrahedron * a,
                                 const aphid::Float4 & coord)
{
	int vx = -1;
	if(coord.x < .03f) vx = a->iv0;
	if(coord.y < .03f) vx = a->iv1;
	if(coord.z < .03f) vx = a->iv2;
	if(coord.w < .03f) vx = a->iv3;
	if(vx < 0) return false;
	
	ITRIANGLE trif = oppositeFace(a, vx);
	
	f->key = aphid::sdb::Coord3(trif.p1, trif.p2, trif.p3);
	f->ta = a;
	int side = findTetrahedronFace(a, &trif);
	f->tb = tetrahedronNeighbor(a, side);
	
	return true;
}

inline void connectTetrahedrons(aphid::sdb::Array<aphid::sdb::Coord3, IFace > & faces)
{
	faces.begin();
	while(!faces.end() ) {
		
		IFace * f = faces.value();
		
		if(f->tb) {
			bool stat = connectTetrahedrons(f->ta, f->tb,
								f->key.x, f->key.y, f->key.z);
			if(!stat) {
				printTetrahedronCannotConnect(f->ta, f->tb);
			}
		}
		
		faces.next();
	}
}

inline void getBoundary(std::vector<IFace *> & boundary,
						aphid::sdb::Array<aphid::sdb::Coord3, IFace > & faces)
{
	faces.begin();
	while(!faces.end() ) {
		
		IFace * f = faces.value();
		
		if(f->tb) {
			
			IFace * tri = new IFace;
			tri->key = f->key;
			tri->ta = f->ta;
			tri->tb = f->tb;
			boundary.push_back(tri);
			
		}
		
		faces.next();
	}
}

inline void addTetrahedronTo(std::vector<ITetrahedron *> & tets,
							ITetrahedron * a)
{
	if(!a) return;
	if(a->index < 0) return;
	std::vector<ITetrahedron *>::iterator it = tets.begin();
	for(;it!= tets.end();++it) {
		if(*it == a) return;
	}
	tets.push_back(a);
}

inline void findTetrahedronAlongEdge(std::vector<ITetrahedron *> & tets,
							const IEdge & e,
							ITetrahedron * a,
							ITetrahedron * b)
{
/// back to origin of loop
	if(a == tets[0])
		return;
		
	addTetrahedronTo(tets, a);
	
	IEdge ea;
	ea.iv0 = e.iv0;
	ea.iv1 = e.iv1;
    findEdgeNeighborPair(&ea, a);
	
	//if(ea.nei0) {
	//	std::cout<<"\n nei0"; printTetrahedronVertices(ea.nei0);
	//}
	//if(ea.nei1) {
	//	std::cout<<"\n nei1"; printTetrahedronVertices(ea.nei1);
	//}
	
	ITetrahedron * c = ea.nei0;
		
/// exclude the parent
	if(!c)
		c = ea.nei1;
		
	if(c == b)
		c = ea.nei1;
		
	if(c)
		findTetrahedronAlongEdge(tets, ea, c, a);

}

inline void findTetrahedronsConnectedToEdge(std::vector<ITetrahedron *> & tets,
							const IEdge & e,
							ITetrahedron * t)
{
	//printTetrahedronVertices(t);
	//std::cout<<"\n edge("<<e.iv0<<", "<<e.iv1<<") ";
	std::cout.flush();
	
    if(e.nei0) {
		findTetrahedronAlongEdge(tets, e, e.nei0, t);
	}
    
    if(e.nei1) {
		findTetrahedronAlongEdge(tets, e, e.nei1, t);
	}
}

inline void expandTetrahedronRegion(std::vector<ITetrahedron *> & tets,
							std::vector<ITetrahedron *> & source)
{
	std::vector<ITetrahedron *>::iterator it = source.begin();
	for(;it!=source.end();++it) {
		addTetrahedronTo(tets, *it );
		addTetrahedronTo(tets, (*it)->nei0);
		addTetrahedronTo(tets, (*it)->nei1);
		addTetrahedronTo(tets, (*it)->nei2);
		addTetrahedronTo(tets, (*it)->nei3);
	}
}

inline void addTetrahedronFaces(ITetrahedron * t,
					aphid::sdb::Array<aphid::sdb::Coord3, IFace > & faces)
{
	ITRIANGLE fa;
	int i=0;
	for(;i<4;++i) {
		faceOfTetrahedron(&fa, t, i);
		aphid::sdb::Coord3 itri = aphid::sdb::Coord3(fa.p1, fa.p2, fa.p3).ordered();
		IFace * tri = faces.find(itri );
		if(!tri) {
			tri = new IFace;
			tri->key = itri;
			tri->ta = t;
			
			faces.insert(itri, tri);
		}
		else {
			tri->tb = t;
		}
	}
}

inline void addTetrahedronFace(ITetrahedron * t,
					const int & va, const int & vb, const int & vc,
					aphid::sdb::Array<aphid::sdb::Coord3, IFace > & faces)
{
	aphid::sdb::Coord3 itri = aphid::sdb::Coord3(va, vb, vc).ordered();
	IFace * tri = faces.find(itri );
	if(tri)
		tri->tb = t;
}

/// two or three tetrahedrons sharing same face (1, 2, 3)
/// and six neighbors
typedef struct {
	ITetrahedron * ta;
	ITetrahedron * tb;
	ITetrahedron * tc;
	int iv0, iv1, iv2, iv3, iv4;
	
} Bipyramid;

inline void resetBipyramid(Bipyramid * pyra)
{ pyra->tb = NULL; }

inline void printBipyramidVertices(const Bipyramid * pyra)
{
	std::cout<<" bipyramid ("<<pyra->iv0<<", "
		<<pyra->iv1<<", "<<pyra->iv2<<", "<<pyra->iv3<<", "
		<<pyra->iv4<<") ";
}

}
#endif