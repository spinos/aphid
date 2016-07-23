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
	m_msh.fillBox(BoundingBox(-40.f, -40.f, -30.f,
								 40.f,  40.f,  30.f), gz);
	
	m_distFunc.addSphere(Vector3F(0.f, 2.8f, 0.f), 25.35f );
	
	//m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 0);
	//m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 1);
	//m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 2);
	//m_msh.subdivideGrid<aphid::BDistanceFunction>(m_distFunc, 3);
	//m_distFunc.addSphere(Vector3F(0.f, -22.43f, 0.2f), 21.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	//m_distFunc.addSphere(Vector3F(0.f, -22200.f, 0.f), 22195.1f );
	
	m_msh.buildGrid();
	m_msh.buildMesh();
	m_msh.buildGraph();
	m_msh.verbose();
	
	m_msh.calculateDistance<BDistanceFunction>(&m_distFunc, 0.f);
	m_msh.estimateError<BDistanceFunction>(&m_distFunc, 0.f, gz * .5f);
	std::cout<<"\n estimated error (min/max) "<<m_msh.minError()<<"/"<<m_msh.maxError();
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
	drawErrors<EdgeRec>(&m_msh, m_msh.dirtyEdges(), 0.067f );
#endif

}

}