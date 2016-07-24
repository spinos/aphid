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
	setColorScale(.5f / gz);
	setNodeDrawSize(gz * .022f);
	m_msh.fillBox(BoundingBox(-50.f, -50.f, -50.f,
								 50.f,  50.f,  50.f), gz);
	
	m_distFunc.addSphere(Vector3F(    7.f, 17.f, -1.f), 26.35f );
	m_distFunc.addSphere(Vector3F(-55.43f, -19.f, 1.f), 49.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	m_distFunc.addSphere(Vector3F(34.f, -10.f, -20.f), 17.1f );
	
#define MAX_BUILD_LEVEL 9
	m_msh.build<BDistanceFunction>(&m_distFunc, MAX_BUILD_LEVEL, 0.06f);
	//m_msh.verbose();
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::draw(aphid::GeoDrawer * dr)
{
	//drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
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