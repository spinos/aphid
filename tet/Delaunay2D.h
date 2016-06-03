/*
 *  Delaunay2D.h
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_DELAUNAY_2D_H
#define TTG_DELAUNAY_2D_H
#include "Scene.h"
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

struct Quadrilateral {
	ITRIANGLE * ta;
	ITRIANGLE * tb;
	ITRIANGLE * nei[4];
	IEDGE e;
	int apex, aapex;
};

class Delaunay2D : public Scene {

	int m_N, m_numTri;
	aphid::Vector3F * m_X;
	ITRIANGLE * m_triangles;
	std::deque<IEDGE> m_edges;
	
public:
	Delaunay2D();
	virtual ~Delaunay2D();
	
	virtual const char * titleStr() const;
	
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	void circumCircle(TriCircle & circ,
						const aphid::Vector3F & p1,
						const aphid::Vector3F & p2,
						const aphid::Vector3F & p3) const;
						
	int searchTri(const aphid::Vector3F & p) const;
	bool insideTri(const aphid::Vector3F & p,
					const aphid::Vector3F & a,
					const aphid::Vector3F & b,
					const aphid::Vector3F & c) const;
	void flipEdges(std::deque<Quadrilateral> & qls);
	bool canEdgeFlip(const Quadrilateral & q) const;
	IEDGE findEdge(const ITRIANGLE * tri, const int & p) const;
	bool findAntiApex(Quadrilateral & q) const;
	int findAntiApex(const ITRIANGLE * tri, const IEDGE & e) const;	
	int findNeighborEdge(const ITRIANGLE * tri,  ITRIANGLE * tgt) const;
	void processEdgeFlip(Quadrilateral & q);
	void findQuadNeighbor(Quadrilateral & q);
	bool connectTriangleAB(ITRIANGLE * b, ITRIANGLE * a, const int & p);
	void printTriangleVertice(const ITRIANGLE * a) const;
};

}
#endif