/*
 *  BccTetraGrid.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_BCC_TET_GRID_H
#define TTG_BCC_TET_GRID_H

#include <WorldGrid.h>
#include <Array.h>

namespace ttg {

class BccNode {
	
public:
	int key;
	int index;
};

class BccCell {

	aphid::Vector3F m_center;
	
	static float TwentySixNeighborOffset[26][3];
	static int SevenNeighborOnCorner[8][7];
	
public:
	BccCell(const aphid::Vector3F &center );
	void addNodes(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::sdb::Coord3 & cellCoord) const;
		
	const aphid::Vector3F * centerP() const;
	
private:
	bool findNeighborCorner(aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > * grid,
					const aphid::Vector3F & pos, int icorner) const;
	int cornerI(const aphid::Vector3F & corner,
				const aphid::Vector3F & center) const;
};

class BccTetraGrid : public aphid::sdb::WorldGrid<aphid::sdb::Array<int, BccNode>, BccNode > 
{
	
public:
	BccTetraGrid();
	virtual ~BccTetraGrid();
	
	void buildTetrahedrons();
	int numNodes();
	
protected:

private:
	void countNodes();
	void countNodesIn(aphid::sdb::Array<int, BccNode> * cell, int & c);
	
};

}

#endif