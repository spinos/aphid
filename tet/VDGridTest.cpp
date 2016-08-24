/*
 *  VDGridTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "VDGridTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

VDGridTest::VDGridTest() 
{}

VDGridTest::~VDGridTest() 
{}
	
const char * VDGridTest::titleStr() const
{ return "View Dependent Grid Test"; }
 
bool VDGridTest::viewPerspective() const
{ return true; }

bool VDGridTest::init()
{
	float gz = 128.f;
	setColorScale(.43f / gz);
	setNodeDrawSize(gz * .008f);
	m_msh.fillBox(BoundingBox(-250.f, -50.f, -250.f,
								 250.f,  50.f,  250.f), gz);
	
	m_distFunc.addSphere(Vector3F(    9.f, 17.f, -1.f), 27.f );
	m_distFunc.addSphere(Vector3F(-54.f, -13.f, -1.f), 64.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	m_distFunc.addSphere(Vector3F(38.f, -10.f, -22.f), 21.1f );
	m_distFunc.addSphere(Vector3F(-100.f, -3420.f, -100.f), 3400.f );
	
#define MAX_BUILD_LEVEL 5
#define MAX_BUILD_ERROR .17f
	m_msh.adaptiveBuild<BDistanceFunction>(&m_distFunc, MAX_BUILD_LEVEL, MAX_BUILD_ERROR);
	
	m_msh.triangulateFront();
	
#if 0
	checkTetraVolumeExt<DistanceNode, ITetrahedron>(m_msh.nodes(), m_msh.numTetrahedrons(),
						m_msh.tetrahedrons() );
#endif

	std::cout.flush();
	return true;
}

void VDGridTest::draw(GeoDrawer * dr)
{
#define SHO_CELL 0
#define SHO_CELL_NODE 0
#define SHO_GRAPH 1
#define SHO_CUT 0
#define SHO_FRONT 1
#define SHO_FRONT_WIRE 0

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
	dr->m_surfaceProfile.apply();
	dr->setColor(0.f, .4f, .5f);
	drawFront<FieldTriangulation >(&m_msh);
		
#if SHO_FRONT_WIRE	
	dr->m_wireProfile.apply();
	dr->setColor(0.1f, .1f, .1f);
	drawFrontWire<FieldTriangulation >(&m_msh);
#endif
#endif

}

void VDGridTest::drawGraph(GeoDrawer * dr)
{
#define SHO_NODE 0
#define SHO_EDGE 0
#define SHO_ERR 1

#if SHO_NODE	
	drawNodes(&m_msh, dr);
#endif

#if SHO_EDGE	
	dr->setColor(0.f, 0.f, .5f);	
	drawEdges(&m_msh, dr);
#endif

#if SHO_ERR
	drawErrors(&m_msh, m_msh.dirtyEdges(), m_msh.errorThreshold() );
#endif

}

void VDGridTest::drawCut(GeoDrawer * dr)
{
	dr->setColor(0.f, 1.f, .6f);
	
	const int & n = m_msh.numAddedVertices();
	for(int i=0; i<n; ++i) {		
		dr->cube(m_msh.addedVertex(i), .2f);
	}
}

}