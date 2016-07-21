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
	int dimx = 12, dimy = 12, dimz = 12;
	float gz = 3.f;
	m_fld.setH(gz);
	m_nodeColScl = 1.f / gz / 8.f;
	m_nodeDrawSize = gz * .0625f;
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
	
	m_distFunc.addSphere(Vector3F(-3.56f, 4.216f, 17.435f), 11.637f );
	m_distFunc.addSphere(Vector3F(0.f, -22.43f, 0.2f), 21.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	//m_distFunc.addSphere(Vector3F(0.f, -22200.f, 0.f), 22195.1f );
	
	m_fld.calculateDistance<BDistanceFunction>(&m_distFunc, 0.1f);
	m_fld.markInsideOutside();
	std::cout<<"\n max estimated error "<<m_fld.estimateError<BDistanceFunction>(&m_distFunc, 0.1f, gz * .5f);
	std::cout.flush();
	return true;
}

void DistanceFieldTest::draw(aphid::GeoDrawer * dr)
{
	drawGraph(dr);
}

void DistanceFieldTest::drawGraph(aphid::GeoDrawer * dr)
{
	int i;
	DistanceNode * v = m_fld.nodes();
	
#define SHO_NODE 1
#define SHO_EDGE 0
#define SHO_ERR 1

#if SHO_NODE	
	dr->setColor(.5f, 0.f, 0.f);
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

#if SHO_ERR
	dr->setColor(0.1f, 0.1f, .1f);
	glBegin(GL_LINES);
	sdb::Array<sdb::Coord2, EdgeRec > * egs = m_fld.dirtyEdges();
	egs->begin();
	while(!egs->end() ) {
		
		if(egs->value()->val > .06f) {
			dr->vertex(v[egs->key().x].pos);
			dr->vertex(v[egs->key().y].pos);
		}
		
		egs->next();
	}
	glEnd();
#endif

}

}