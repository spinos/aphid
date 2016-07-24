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
	
	m_distFunc.addSphere(Vector3F(    8.f, 17.f, -1.f), 23.35f );
	m_distFunc.addSphere(Vector3F(-55.43f, -19.f, 1.f), 60.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	m_distFunc.addSphere(Vector3F(33.f, -11.f, -22.f), 19.1f );
	
#define MAX_BUILD_LEVEL 4
	m_msh.build<BDistanceFunction>(&m_distFunc, MAX_BUILD_LEVEL, .0313f);
	
#if 0
	checkTetraVolumeExt<DistanceNode, ITetrahedron>(m_msh.nodes(), m_msh.numTetrahedrons(),
						m_msh.tetrahedrons() );
#endif
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::draw(aphid::GeoDrawer * dr)
{
#define SHO_CELL 0
#define SHO_CELL_NODE 0
#define SHO_GRAPH 1

#if SHO_CELL
	drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
#endif

#if SHO_CELL_NODE
	drawGridNode<AdaptiveBccGrid3, BccCell3>(m_msh.grid(), dr);
#endif

#if SHO_GRAPH
	drawGraph(dr);
#endif
}

void AdaptiveGridTest::drawGraph(aphid::GeoDrawer * dr)
{
#define SHO_NODE 1
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
	dr->setColor(0.1f, 0.1f, .1f);
	drawErrors<EdgeRec>(&m_msh, m_msh.dirtyEdges(), .0313f );
#endif

}

}