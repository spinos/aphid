/*
 *  AdaptiveGridTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptiveGridTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

AdaptiveGridTest::AdaptiveGridTest() 
{}

AdaptiveGridTest::~AdaptiveGridTest() 
{}
	
const char * AdaptiveGridTest::titleStr() const
{ return "Adaptive Grid Test"; }

bool AdaptiveGridTest::init()
{
	float gz = 32.f;
	m_nodeColScl = gz / .02f;
	m_nodeDrawSize = gz * .022f;
	m_msh.fillBox(BoundingBox(-30.f, -30.f, -30.f,
								 30.f,  30.f,  30.f), gz);
	
	m_distFunc.addSphere(Vector3F(0.001f, 0.f, 0.f), 23.5f );
	
	m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 0);
	m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 1);
	m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 2);
	//m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 3);
	//m_distFunc.addSphere(Vector3F(0.f, -22.43f, 0.2f), 21.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	//m_distFunc.addSphere(Vector3F(0.f, -22200.f, 0.f), 22195.1f );
	
	//m_msh.calculateDistance<BDistanceFunction>(&m_distFunc, 0.1f);
	//m_msh.markInsideOutside();
	//std::cout<<"\n max estimated error "<<m_msh.estimateError<BDistanceFunction>(&m_distFunc, 0.1f, gz * .5f);
	m_msh.buildGrid();
	m_msh.buildMesh();
	m_msh.buildGraph();
	m_msh.verbose();
	std::cout<<"\n n tetra "<<m_msh.numTetrahedrons();
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::draw(aphid::GeoDrawer * dr)
{
	drawGrid(dr);
	//drawGraph(dr);
}

void AdaptiveGridTest::drawGrid(aphid::GeoDrawer * dr)
{
	dr->setColor(.15f, .15f, .15f);
	
	AdaptiveBccGrid3 * grd = m_msh.grid();
	dr->boundingBox(grd->boundingBox() );
	
	Vector3F cellCol;
	BoundingBox cellBox;
	grd->begin();
	while(!grd->end() ) {
		
		sdb::gdt::GetCellColor(cellCol, grd->key().w );
		grd->getCellBBox(cellBox, grd->key() );
		cellBox.expand(-.04f - .04f * grd->key().w );
		
		dr->setColor(cellCol.x, cellCol.y, cellCol.z);
		dr->boundingBox(cellBox);
		
		//drawNode(grd->value(), dr, grd->key().w );
		
		grd->next();
	}
}

void AdaptiveGridTest::drawNode(BccCell3 * cell, aphid::GeoDrawer * dr,
								const float & level)
{
	float r, g, b;
	float nz = m_nodeDrawSize * (1.f - .12f * level);
	
	cell->begin();
	while(!cell->end() ) {
		BccCell::GetNodeColor(r, g, b,
					cell->value()->prop);
		dr->setColor(r, g, b);
		dr->cube(cell->value()->pos, nz );
		
		cell->next();
	}
}

void AdaptiveGridTest::drawGraph(aphid::GeoDrawer * dr)
{
	int i;
	DistanceNode * v = m_msh.nodes();
	
#define SHO_NODE 0
#define SHO_EDGE 1
#define SHO_ERR 0

#if SHO_NODE	
	dr->setColor(.5f, 0.f, 0.f);
	const int nv = m_msh.numNodes();

	Vector3F col;
	for(i = 0;i<nv;++i) {
		const DistanceNode & vi = v[i];
		
		m_msh.nodeColor(col, vi, m_nodeColScl);
		dr->setColor(col.x, col.y, col.z);
		dr->cube(vi.pos, m_nodeDrawSize);
		
	}
#endif

#if SHO_EDGE	
	dr->setColor(0.f, 0.f, .5f);
	IGraphEdge * e = m_msh.edges();
	const int ne = m_msh.numEdges();
	
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
	sdb::Array<sdb::Coord2, EdgeRec > * egs = m_msh.dirtyEdges();
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