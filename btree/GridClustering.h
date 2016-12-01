/*
 *  GridClustering.h
 *  
 *
 *  Created by jian zhang on 3/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  clustering by quantizing element enter
 */

#pragma once
#include <sdb/WorldGrid.h>
#include <BoundingBox.h>
#include <sdb/VectorArray.h>

namespace aphid {

class GroupCell : public sdb::Sequence<unsigned>
{
public:
	GroupCell(sdb::Entity * parent = NULL) : sdb::Sequence<unsigned>(parent) {}
	BoundingBox m_box;
};

class GridClustering : public sdb::WorldGrid<GroupCell, unsigned >
{

public :
	GridClustering();
	virtual ~GridClustering();
	
	void insertToGroup(const BoundingBox & box, const unsigned & idx);
	unsigned numElements();
	void extractInside(sdb::VectorArray<unsigned> & dst, const sdb::VectorArray<BoundingBox> & boxSrc,
					const BoundingBox & inBox);
	
protected :

private:
	void extractCellInside(GroupCell * cell, sdb::VectorArray<unsigned> & dst, const sdb::VectorArray<BoundingBox> & boxSrc,
					const BoundingBox & inBox);
					
};

}