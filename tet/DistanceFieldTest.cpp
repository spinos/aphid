/*
 *  DistanceFieldTest.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DistanceFieldTest.h"
#include <iostream>

using namespace aphid;
namespace ttg {

DistanceFieldTest::DistanceFieldTest() 
{ std::cout<<"\n init distance field "; }

DistanceFieldTest::~DistanceFieldTest() 
{}
	
const char * DistanceFieldTest::titleStr() const
{ return "Distance Field Test"; }

bool DistanceFieldTest::init()
{
	int i, j, k;
	int dimx = 6, dimy = 6, dimz = 6;
	float gz = 4.f;
	m_gridmk.setH(gz);
	Vector3F ori(gz*.5f, gz*.5f, gz*.5f);
	std::cout<<"\n cell size "<<gz
		<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
	for(k=0; k<dimz;++k) {
		for(j=0; j<dimy;++j) {
			for(i=0; i<dimx;++i) {
				m_gridmk.addCell(ori + Vector3F(i, j, k) * gz );
			}
		}
	}
	m_gridmk.buildGrid();
	m_gridmk.buildMesh();
	m_gridmk.buildGraph();
	std::cout<<"\n grid n tetra "<<m_gridmk.numTetrahedrons()
		<<"\n grid n node "<<m_gridmk.numNodes()
		<<"\n grid n edge "<<m_gridmk.numEdges()
		<<"\n grid n edge ind "<<m_gridmk.numEdgeIndices();
	std::cout.flush();
	return true;
}

void DistanceFieldTest::draw(aphid::GeoDrawer * dr)
{
}

}