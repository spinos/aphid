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
	
	m_nodeColScl = gz / .02f;
	m_nodeDrawSize = gz * .022f;
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
	
	m_distFunc.addSphere(Vector3F(0.f, 1.421f, 0.f), 8.437f );
	
	subdivideGrid(0);
	subdivideGrid(1);
	//subdivideGrid(2);
	//subdivideGrid(3);
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
	m_grd.build();
	std::cout.flush();
	return true;
}

void AdaptiveGridTest::subdivideGrid(int level)
{
/// track cells divided
	std::vector<sdb::Coord4 > divided;
	
	BoundingBox cellBox;
	
	m_grd.begin();
	while(!m_grd.end() ) {
		
		if(m_grd.key().w == level) {
			m_grd.getCellBBox(cellBox, m_grd.key() );
			//std::cout<<"\n cell box "<<cellBox;
			
			if(m_distFunc.intersect<BoundingBox >(&cellBox) ) {
				
				m_grd.subdivideCell(m_grd.key() );
				
				divided.push_back(m_grd.key() );
			}
		}
		
		if(m_grd.key().w > level)
			break;
			
		m_grd.next();
	}
	
	if(level > 1)
		enforceBoundary(divided);
		
	divided.clear();
}

void AdaptiveGridTest::enforceBoundary(const std::vector<sdb::Coord4 > & ks)
{
	std::vector<sdb::Coord4 >::const_iterator it = ks.begin();
	for(;it!=ks.end();++it) {
		
		for(int i=0; i< 6;++i) {
			const sdb::Coord4 nei = m_grd.neighborCoord(*it, i);

			if(!m_grd.findCell(nei) ) {
				const sdb::Coord4 par = m_grd.parentCoord(nei);
				m_grd.subdivideCell(par );
			}
		}
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
		
		m_grd.GetCellColor(cellCol, m_grd.key().w );
		m_grd.getCellBBox(cellBox, m_grd.key() );
		//cellBox.expand(-.04f - .04f * m_grd.key().w );
		
		dr->setColor(cellCol.x, cellCol.y, cellCol.z);
		dr->boundingBox(cellBox);
		
		drawNode(m_grd.value(), dr );
		
		m_grd.next();
	}
}

void AdaptiveGridTest::drawNode(BccCell3 * cell, aphid::GeoDrawer * dr)
{
	float r, g, b;
	
	cell->begin();
	while(!cell->end() ) {
		BccCell::GetNodeColor(r, g, b,
					cell->value()->prop);
		dr->setColor(r, g, b);
		dr->cube(cell->value()->pos, m_nodeDrawSize);
		
		cell->next();
	}
}

}