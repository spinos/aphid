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
	const float gz = tb.getLongestDistance() * .67f;
	const Vector3F cent = tb.center();
	tb.setMin(cent.x - gz, cent.y - gz, cent.z - gz );
	tb.setMax(cent.x + gz, cent.y + gz, cent.z + gz );
	setColorScale(.25f / gz);
	setNodeDrawSize(gz * .016f);
	
	m_msh.fillBox(tb, gz);
	
	m_distFunc.addTree(m_container.tree() );
	
#define MAX_BUILD_LEVEL 1
#define MAX_BUILD_ERROR_FAC .0078125f /// 1/128
//#define MAX_BUILD_ERROR_FAC .015625f /// 1/64
	//m_msh.build<BDistanceFunction>(&m_distFunc, MAX_BUILD_LEVEL, gz * MAX_BUILD_ERROR_FAC);
	m_msh.discretize<BDistanceFunction>(&m_distFunc, 6, gz * MAX_BUILD_ERROR_FAC );
	
	m_msh.grid()->calculateBBox();
	m_msh.grid()->build();
	
	std::cout<<"\n grid n cell "<<m_msh.grid()->size()
			<<"\n grid bbx "<<m_msh.grid()->boundingBox();
	std::cout.flush();
	return true;
}

void KDistanceTest::draw(GeoDrawer * dr)
{
#define SHO_TREE 0
#define SHO_CELL 1

#if SHO_TREE
	drawTree(dr);
#endif

#if SHO_CELL
	drawGridCell<AdaptiveBccGrid3>(m_msh.grid(), dr);
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