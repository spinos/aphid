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
    
template<typename T>
class SparseMatrix;

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
/// left-hand-side matrix of global solver 
/// M/h^2 + sigma(w_i S_i^T A_i^T A_i S_i)
/// w is weight or stiffness of constraint_i
/// S_i is selection matrix that selects the vertices involved in the i-th constraint
/// only involved particle columns are one
/// attachment S 3-by-3n n is num of particle
/// spring S 6-by-3n
/// A_i is constraint matrix
/// attachment A 3-by-3
/// |1    |
/// |  1  |
/// |    1|
/// spring A 6-by-6
/// | 0.5        -0.5          |
/// |     0.5         -0.5     |
/// |         0.5          -0.5|
/// |-0.5         0.5          |
/// |    -0.5          0.5     |
/// |        -0.5           0.5|
	SparseMatrix<float> * m_lhsMat;
/// each constraint has its lhs
/// w_i S_i^T A_i^T B_i
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

