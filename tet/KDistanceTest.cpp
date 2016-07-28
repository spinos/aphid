/*
 *  KDistanceTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "KDistanceTest.h"
#include <NTreeDrawer.h>
#include <iostream>

using namespace aphid;
namespace ttg {

KDistanceTest::KDistanceTest(const std::string & filename) 
{
	m_fileName = filename;
}

KDistanceTest::~KDistanceTest() 
{}
	
const char * KDistanceTest::titleStr() const
{ return "Kd-Tree + Adaptive Grid Test"; }

bool KDistanceTest::init()
{
/// no grid
	m_container.readTree(m_fileName, 0);
	
	BoundingBox tb = m_container.tree()->getBBox();
	const float gz = tb.getLongestDistance() * .53f;
	const Vector3F cent = tb.center();
	tb.setMin(cent.x - gz, cent.y - gz, cent.z - gz );
	tb.setMax(cent.x + gz, cent.y + gz, cent.z + gz );
	setColorScale(.08f / gz);
	setNodeDrawSize(gz * GDT_FAC_ONEOVER32 );
	
	m_msh.fillBox(tb, gz);
	
	m_distFunc.addTree(m_container.tree() );
	
/// grid is very coarse relative to input mesh, error will be large
/// just subdivide to level ignoring error
	m_msh.discretize<BDistanceFunction>(&m_distFunc, 4, gz * GDT_FAC_ONEOVER16 );
	
	m_msh.buildGrid();
	m_msh.buildMesh();
	m_msh.buildGraph();
	std::cout<<"\n grid n cell "<<m_msh.grid()->size()
			<<"\n grid bbx "<<m_msh.grid()->boundingBox()
			<<"\n n node "<<m_msh.numNodes()
			<<"\n n edge "<<m_msh.numEdges();
	m_distFunc.setDomainDistanceRange(gz * GDT_FAC_ONEOVER16 * 1.9f );
	m_msh.calculateDistance<BDistanceFunction>(&m_distFunc, gz * GDT_FAC_ONEOVER16);
	m_msh.triangulateFront();
	
	std::cout.flush();
	return true;
}

void KDistanceTest::draw(GeoDrawer * dr)
{
#define SHO_TREE 0
#define SHO_CELL 0
#define SHO_NODE 0
#define SHO_EDGE 0
#define SHO_ERR 0
#define SHO_FRONT 1
#define SHO_FRONT_WIRE 0

#if SHO_TREE
	drawTree(dr);
#endif

#if SHO_CELL
	drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
#endif

#if SHO_NODE
	drawNodes(&m_msh, dr);
#endif

#if SHO_EDGE	
	dr->setColor(0.f, 0.f, .5f);
	drawEdges(&m_msh, dr);
#endif

#if SHO_FRONT	
	dr->m_surfaceProfile.apply();
	dr->setColor(0.f, .4f, .5f);
	drawFront<FieldTriangulation >(&m_msh);
#endif
		
#if SHO_FRONT_WIRE	
	dr->m_wireProfile.apply();
	dr->setColor(0.1f, .1f, .1f);
	drawFrontWire<FieldTriangulation >(&m_msh);
#endif

}

void KDistanceTest::drawTree(aphid::GeoDrawer * dr)
{
	if(!m_container.tree() ) return; 
	
	dr->m_wireProfile.apply();
	dr->setColor(.15f, .25f, .35f);
	dr->boundingBox(m_container.tree()->getBBox() );
	
	NTreeDrawer tdr;
	tdr.drawTree<cvx::Triangle>(m_container.tree() );
}

}