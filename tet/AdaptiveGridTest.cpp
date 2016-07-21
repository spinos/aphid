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
	m_grd.setFinestCellSize(1.f);
	
	int i, j, k;
	int dimx = 2, dimy = 2, dimz = 2;
	float gz = m_grd.coarsestCellSize();
	
	m_nodeColScl = gz / .03125f;
	m_nodeDrawSize = gz * .032f;
	Vector3F ori(gz*.5f - gz*dimx/2, 
					gz*.5f - gz*dimy/2, 
					gz*.5f - gz*dimz/2);
	std::cout<<"\n level0 cell size "<<gz
		<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
		
	for(k=0; k<dimz;++k) {
		for(j=0; j<dimy;++j) {
			for(i=0; i<dimx;++i) {
				m_grd.addCell(ori + Vector3F(i, j, k) * gz );
			}
		}
	}	
	
	m_distFunc.addSphere(Vector3F(0.f, 0.21f, 0.f), 9.637f );
	
	subdivideGrid(0);
	subdivideGrid(1);
	subdivideGrid(2);
	//m_distFunc.addSphere(Vector3F(0.f, -22.43f, 0.2f), 21.f );
	//m_distFunc.addBox(Vector3F(-40.f, -12.f, -10.f),
	//					Vector3F(40.f, -7.87f, 40.f) );
	//m_distFunc.addSphere(Vector3F(0.f, -22200.f, 0.f), 22195.1f );
	
	//m_fld.calculateDistance<BDistanceFunction>(&m_distFunc, 0.1f);
	//m_fld.markInsideOutside();
	//std::cout<<"\n max estimated error "<<m_fld.estimateError<BDistanceFunction>(&m_distFunc, 0.1f, gz * .5f);
	m_grd.calculateBBox();
	std::cout<<"\n grid bbox "<<m_grd.boundingBox()
			<<"\n grid n cell "<<m_grd.size();
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::subdivideGrid(int level)
{
	BoundingBox cellBox;
	
	m_grd.begin();
	while(!m_grd.end() ) {
		
		if(m_grd.key().w == level) {
			m_grd.getCellBBox(cellBox, m_grd.key() );
			//std::cout<<"\n cell box "<<cellBox;
			
			if(m_distFunc.intersect<BoundingBox >(&cellBox) ) {
				
				for(int i=0;i<8;++i) {
					m_grd.getCellChildBox(cellBox, i, m_grd.key() );
					//std::cout<<"\n child box"<<i<<" "<<cellBox;
					if(m_distFunc.intersect<BoundingBox >(&cellBox) ) {
						m_grd.subdivide(m_grd.key(), i);
					}
				}
			}
		}
		
		if(m_grd.key().w > level)
			break;
			
		m_grd.next();
	}
}

void AdaptiveGridTest::draw(aphid::GeoDrawer * dr)
{
	drawGrid(dr);
}

void AdaptiveGridTest::drawGrid(aphid::GeoDrawer * dr)
{
	dr->setColor(.15f, .15f, .15f);
	dr->boundingBox(m_grd.boundingBox() );
	
	Vector3F cellCol;
	BoundingBox cellBox;
	m_grd.begin();
	while(!m_grd.end() ) {
		
		m_grd.getCellColor(cellCol, m_grd.key().w );
		m_grd.getCellBBox(cellBox, m_grd.key() );
		cellBox.expand(-.02f - .02f * m_grd.key().w );
		
		dr->setColor(cellCol.x, cellCol.y, cellCol.z);
		dr->boundingBox(cellBox);
		
		m_grd.next();
	}
}

}