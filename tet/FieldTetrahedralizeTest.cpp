/*
 *  FieldTetrahedralizeTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "FieldTetrahedralizeTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

FieldTetrahedralizeTest::FieldTetrahedralizeTest() 
{}

FieldTetrahedralizeTest::~FieldTetrahedralizeTest() 
{}
	
const char * FieldTetrahedralizeTest::titleStr() const
{ return "Distance Field Tetrahedronize Test"; }

bool FieldTetrahedralizeTest::init()
{
	int i, j, k;
	int dimx = 12, dimy = 10, dimz = 12;
	float gz = 2.97f;
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
	
	m_distFunc.addSphere(Vector3F(-3.56f, 1.263f, 18.735f), 11.637f );
	m_distFunc.addBox(Vector3F(-40.f, -9.3f, -10.f),
						Vector3F(40.f, -5.43125f, 40.f) );
	
	m_fld.calculateDistance<BDistanceFunction>(&m_distFunc);
	m_fld.markInsideOutside();
	m_fld.buildRefinedMesh();
	m_fld.checkTetraVolume();
	std::cout.flush();
		
	return true;
}

void FieldTetrahedralizeTest::draw(aphid::GeoDrawer * dr)
{
#define SHO_GRAPH 0
#define SHO_GRID 1

#if SHO_GRAPH
	drawGraph(dr);
#endif
	
#if SHO_GRID
	drawGrid(dr);
#endif
}

void FieldTetrahedralizeTest::drawGraph(aphid::GeoDrawer * dr)
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

void FieldTetrahedralizeTest::drawGrid(aphid::GeoDrawer * dr)
{
	float r, g, b;
	BccTetraGrid * grd = m_fld.grid();
	grd->begin();
	while(!grd->end() ) {
		
		sdb::Array<int, BccNode> * cell = grd->value();
		
		cell->begin();
		while(!cell->end() ) {
		
			const BccNode * n = cell->value();
		
			if(n->prop > 0) {
				BccCell::GetNodeColor(r, g, b, n->prop);
				dr->setColor(r, g, b);
				dr->cube(n->pos, m_nodeDrawSize * 1.2f);
			}
			
			cell->next();
		}
		
		grd->next();
	}

}

}