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
	setColorScale(.4f / gz);
	setNodeDrawSize(gz * .022f);
	m_msh.fillBox(BoundingBox(-50.f, -50.f, -50.f,
								 50.f,  50.f,  50.f), gz);
	
	m_distFunc.addSphere(Vector3F(    9.f, 17.f, -1.f), 24.35f );
	m_distFunc.addSphere(Vector3F(-55.43f, -19.f, 1.f), 62.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	m_distFunc.addSphere(Vector3F(33.f, -11.f, -22.f), 22.1f );
	
#define MAX_BUILD_LEVEL 5
	m_msh.build<BDistanceFunction>(&m_distFunc, MAX_BUILD_LEVEL, .04f);
	
	m_msh.triangulateFront();
	
#if 0
	checkTetraVolumeExt<DistanceNode, ITetrahedron>(m_msh.nodes(), m_msh.numTetrahedrons(),
						m_msh.tetrahedrons() );
#endif

	std::cout.flush();
	return true;
}

void AdaptiveGridTest::draw(GeoDrawer * dr)
{
#define SHO_CELL 0
#define SHO_CELL_NODE 0
#define SHO_GRAPH 1
#define SHO_CUT 0
#define SHO_FRONT 1

#if SHO_CELL
	drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
#endif

#if SHO_CELL_NODE
	drawGridNode<AdaptiveBccGrid3, BccCell3>(m_msh.grid(), dr);
#endif

#if SHO_GRAPH
	drawGraph(dr);
#endif

#if SHO_CUT
	drawCut(dr);
#endif

#if SHO_FRONT
	drawFront(dr);
#endif
}

void AdaptiveGridTest::drawGraph(GeoDrawer * dr)
{
#define SHO_NODE 1
#define SHO_EDGE 1
#define SHO_ERR 1

#if SHO_NODE	
	drawNodes(&m_msh, dr);
#endif

#if SHO_EDGE	
	dr->setColor(0.f, 0.f, .5f);	
	drawEdges(&m_msh, dr);
#endif

#if SHO_ERR
	dr->setColor(0.1f, 0.1f, .1f);
	drawErrors<EdgeRec>(&m_msh, m_msh.dirtyEdges(), .04f );
#endif

}

void AdaptiveGridTest::drawCut(GeoDrawer * dr)
{
	dr->setColor(0.f, 1.f, .6f);
	
	const int & n = m_msh.numAddedVertices();
	for(int i=0; i<n; ++i) {		
		dr->cube(m_msh.addedVertex(i), .2f);
	}
}

void AdaptiveGridTest::drawFront(GeoDrawer * dr)
{
	
	
#if 1
	const int & n = m_msh.numFrontTriangles();
#else
	const int & n = m_msh.numTetrahedrons();
#endif

	Vector3F p1, p2, p3;
	dr->setColor(0.f, .4f, .5f);
	glBegin(GL_TRIANGLES);
	for(int i=0; i<n; ++i) {
		p1 = m_msh.triangleP(i, 0);
		p2 = m_msh.triangleP(i, 1);
		p3 = m_msh.triangleP(i, 2);
		
		glVertex3fv((const float *)&p1);
		glVertex3fv((const float *)&p2);
		glVertex3fv((const float *)&p3);
	}
	glEnd();
	
	dr->setColor(0.1f, .1f, .1f);
	glBegin(GL_LINES);
	for(int i=0; i<n; ++i) {		
		
#if 1
		p1 = m_msh.triangleP(i, 0);
		p2 = m_msh.triangleP(i, 1);
		p3 = m_msh.triangleP(i, 2);
		
		glVertex3fv((const float *)&p1);
		glVertex3fv((const float *)&p2);
		
		glVertex3fv((const float *)&p2);
		glVertex3fv((const float *)&p3);
		
		glVertex3fv((const float *)&p3);
		glVertex3fv((const float *)&p1);
		
#else
		const ITetrahedron * t = m_msh.tetrahedrons()[i];
		p1 = m_msh.nodes()[t->iv0].pos;
		p2 = m_msh.nodes()[t->iv1].pos;
		p3 = m_msh.nodes()[t->iv2].pos;
		p4 = m_msh.nodes()[t->iv3].pos;
		
		glVertex3fv((const float *)&p1);
		glVertex3fv((const float *)&p2);
		
		glVertex3fv((const float *)&p2);
		glVertex3fv((const float *)&p3);
		
		glVertex3fv((const float *)&p3);
		glVertex3fv((const float *)&p1);
		
		glVertex3fv((const float *)&p4);
		glVertex3fv((const float *)&p1);
#endif
		
	}
	glEnd();
}

}