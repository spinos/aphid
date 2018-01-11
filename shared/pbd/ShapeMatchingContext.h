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
#include "ShapeMatchingRegion.h"

namespace aphid {
    
template<typename T>
class SparseMatrix;

template<typename T>
class ConjugateGradient;

namespace sdb {
class Coord2;
}

namespace pbd {

class ShapeMatchingProfile;

class ShapeMatchingContext : public SimulationContext {

/// left-hand-side stiffness matrix K 3n-by-3n
/// K <- M/h^2 + sigma(w_i S_i^T A_i^T A_i S_i)
/// w_i is weight or stiffness of constraint_i
/// S_i is selection matrix that selects the vertices involved in the i-th constraint
/// only involved particle columns are one
/// A_i is constraint matrix
	SparseMatrix<float> * m_lhsMat;
/// linear system solver
	ConjugateGradient<float> * m_cg;
	std::vector<ShapeMatchingRegion* > m_regions;
/// k
	float m_stiffness;
	
public:
    
    ShapeMatchingContext();
	virtual ~ShapeMatchingContext();

	void create(const ShapeMatchingProfile& prof);
	void setStiffness(const float& x);
	
	int numRegions() const;
	const ShapeMatchingRegion* region(const int& i) const;
	
protected:
	virtual void updateShapeMatchingRegions();
	virtual void positionConstraintProjection();

private:
	void clearConstraints();
	
};

}
}
#endif

