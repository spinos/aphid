/*
 *  GridClustering.cpp
 *  
 *
 *  Created by jian zhang on 3/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridClustering.h"
#include <iostream>

namespace aphid {

GridClustering::GridClustering() {}
GridClustering::~GridClustering() {}

void GridClustering::insertToGroup(const BoundingBox & box, const unsigned & idx)
{
	const Vector3F center = box.center();
	
	GroupCell * c = insertChild((const float *)&center);
		
	if(!c) {
		std::cout<<"\n error cast to GroupCell";
		return;
	}
	
	c->insert(idx);
	c->m_box.expandBy(box);
}

unsigned GridClustering::numElements()
{
	unsigned sum = 0;
	begin();
	while(!end() ) {
		sum += value()->size();
		next();
	}
	return sum;
}

void GridClustering::extractInside(sdb::VectorArray<unsigned> & dst, const sdb::VectorArray<BoundingBox> & boxSrc,
					const BoundingBox & inBox)
{
	begin();
	while(!end() ) {
		extractCellInside(value(), dst, boxSrc, inBox);
		next();
	}
}

void GridClustering::extractCellInside(GroupCell * cell, sdb::VectorArray<unsigned> & dst, const sdb::VectorArray<BoundingBox> & boxSrc,
					const BoundingBox & inBox)
{
	cell->begin();
	while(!cell->end() ) {
		const BoundingBox * primB = boxSrc[cell->key() ];
		if(primB->touch(inBox) )
			dst.insert(cell->key() );
		cell->next();
	}
}

}