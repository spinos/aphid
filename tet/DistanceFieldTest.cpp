/*
 *  DistanceFieldTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DistanceFieldTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

DistanceFieldTest::DistanceFieldTest() 
{}

DistanceFieldTest::~DistanceFieldTest() 
{}
	
const char * DistanceFieldTest::titleStr() const
{ return "Distance Field Test"; }

bool DistanceFieldTest::init()
{
	int i, j, k;
	int dimx = 10, dimy = 8, dimz = 9;
	float gz = 3.13f;
	m_fld.setH(gz);
	m_nodeColScl = 1.f / gz / 4;
	m_nodeDrawSize = gz * .125f;
	Vector3F ori(gz*.5f - gz*dimx/2, 
					gz*.5f - gz*dimy/2, 
					gz*.5f);
	std::cout<<"\n cell size "<<gz
		<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
	for(k=0; k<dimz;++k) {
		for(j=0; j<dimy;++j) {
			for(i=0; i<dimx;++i) {
				m_fld.addCell(ori + Vector3F(i, j, k) * gz );
			}
		}
	}
	m_fld.buildGrid();
	m_fld.buildMesh();
	m_fld.buildGraph();
	m_fld.verbose();
	
	m_distFunc.addSphere(Vector3F(-2.56f, -3.46f, 17.435f), 9.737f );
	
	m_fld.calculateDistance<BDistanceFunction>(&m_distFunc);
	
	return true;
}

void DistanceFieldTest::draw(aphid::GeoDrawer * dr)
{
	drawGraph(dr);
}

void DistanceFieldTest::drawGraph(aphid::GeoDrawer * dr)
{
	int i;
#define SHO_NODE 1
#define SHO_EDGE 0

#if SHO_NODE	
	dr->setColor(.5f, 0.f, 0.f);
	DistanceNode * v = m_fld.nodes();
	const int nv = m_fld.numNodes();

	Vector3F col;
	for(i = 0;i<nv;++i) {
		const DistanceNode & vi = v[i];
		
		m_fld.nodeColor(col, vi, m_nodeColScl);
		dr->setColor(col.x, col.y, col.z);
		dr->cube(vi.pos, m_nodeDrawSize);
		
	}
#endif

#if SHO_EDGE	
	dr->setColor(0.f, 0.f, .5f);
	IGraphEdge * e = m_fld.edges();
	const int ne = m_fld.numEdges();
	
	glBegin(GL_LINES);
	for(i = 0;i<ne;++i) {
		const IGraphEdge & ei = e[i];
		
		dr->vertex(v[ei.vi.x].pos);
		dr->vertex(v[ei.vi.y].pos);
		
	}
	glEnd();
#endif

	
}

}