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
	
	const BoundingBox tb = m_container.tree()->getBBox();
	const float gz = tb.getLongestDistance() * .53f;
	setColorScale(.25f / gz);
	setNodeDrawSize(gz * .016f);
	m_msh.fillBox(tb, gz);
	
	
	std::cout.flush();
	return true;
}

void KDistanceTest::draw(GeoDrawer * dr)
{
#define SHO_TREE 1

#if SHO_TREE
	drawTree(dr);
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