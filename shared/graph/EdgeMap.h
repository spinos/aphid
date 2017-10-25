/*
 *  EdgeMap.h
 *  
 *  array of ints with 2d key
 *  edge.v0 < edge.v1
 *
 *  Created by jian zhang on 10/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GRH_EDGE_MAP_H
#define APH_GRH_EDGE_MAP_H

#include <sdb/Array.h>
#include <vector>

namespace aphid {

namespace grh {

class EdgeMap : public sdb::Array<sdb::Coord2, int> {

public:
	EdgeMap();
	virtual ~EdgeMap();
	
	void createFromTriangles(const int& triangleCount,
				const int* triangleIndices);
	
	void buildVertexVaryingEdges(std::vector<int>& edgeBegins,
				std::vector<int>& edgeInds);

	int * findEdge(const int & v1, const int & v2);
    void resetIndices();
    
protected:

private:
	void addEdge(const sdb::Coord2 & e);
    void mergeIndices(const std::vector<int> & a,
							std::vector<int> & b) const;
							
};

}

}

#endif
