/*
 *  GridMaker.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_GRID_MAKER_H
#define TTG_GRID_MAKER_H

#include "BccTetraGrid.h"

namespace ttg {

class GridMaker {

	BccTetraGrid m_grid;
	aphid::sdb::Sequence<aphid::sdb::Coord2> m_graphEdges;
	std::vector<ITetrahedron *> m_tets;
	std::vector<int> m_vvEdgeBegins;
	std::vector<int> m_vvEdgeInds;
	
public:
	GridMaker();
	virtual ~GridMaker();
	
	void setH(const float & x);
	void addCell(const aphid::Vector3F & p);
	void buildGrid();
	void buildMesh();
	void buildGraph();
	int numNodes();
	int numEdges();
	int numTetrahedrons() const;
	int numEdgeIndices() const;
	
protected:
	BccTetraGrid * grid();
	
private:
	void pushIndices(const std::vector<int> & a);
	
};

}
#endif