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
	setColorScale(1.f / gz);
	setNodeDrawSize(gz * .022f);
	m_msh.fillBox(BoundingBox(-40.f, -40.f, -40.f,
								 40.f,  40.f,  40.f), gz);
	
	m_distFunc.addSphere(Vector3F(    10.f, 1.f, 0.f), 25.35f );
	m_distFunc.addSphere(Vector3F(-120.43f, -1.f, 0.f), 111.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	//m_distFunc.addSphere(Vector3F(0.f, -22200.f, 0.f), 22195.1f );
	
	m_msh.build<BDistanceFunction>(&m_distFunc, 3, .06f);
	//m_msh.verbose();
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::draw(aphid::GeoDrawer * dr)
{
	drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
	//drawGridNode<AdaptiveBccGrid3, BccCell3>(m_msh.grid(), dr);
	drawGraph(dr);
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
	drawErrors<EdgeRec>(&m_msh, m_msh.dirtyEdges(), 0.06f );
#endif

}

}