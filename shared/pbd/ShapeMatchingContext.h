/*
 *  ShapeMatchingContext.h
 *  
 *  strands (two parallel lines) with shape-matching constraints
 *
 *  Created by jian zhang on 1/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_SHAPE_MATCHING_CONTEXT_H
#define APH_PBD_SHAPE_MATCHING_CONTEXT_H

#include "SimulationContext.h"

namespace aphid {

namespace sdb {
class Coord2;
}

namespace pbd {

class ShapeMatchingContext : public SimulationContext {

	int m_numStrands;
/// for each strand
	int* m_strandBegin;
	int m_numEdges;
	sdb::Coord2* m_edgeInds;
	float* m_restEdgeLs;
	
public:
	ShapeMatchingContext();
	virtual ~ShapeMatchingContext();
/// each strand has 
/// 2n vertices
/// n-1 segments
/// n-2 regions
/// - v0 - v3 -
///   |    |
/// - v1 - v2 -
	void create();
	
	const int& numEdges() const;
	void getEdge(int& v1, int& v2, const int& i) const;
	
protected:

};

}
}
#endif
