/*
 *  ShapeMatchingRegion.h
 *  
 *  region with limited number of vertices and edges
 *
 *  Created by jian zhang on 1/13/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_SHAPE_MATCHING_REGION_H
#define APH_PBD_SHAPE_MATCHING_REGION_H

#include <math/Quaternion.h>

namespace aphid {

namespace pbd {

struct RegionVE {
	int* _vinds;
	int* _einds;
	int _nv;
	int _ne;
};

class ShapeMatchingRegion {

/// rotation
	Quaternion m_rotq;
/// indices to p
	int m_ind[8];
/// indices to edge v
	int m_edge[8][2];
/// edge v to g_i
	int m_goalInd[8][2];
/// x_i^0 - x_cm^0
	float m_q[3][8];
/// x_i - x_cm
	float m_p[3][8];
/// g_i goal positions
	float m_g[8][3];
/// m_i
	float m_mass[8];
	float m_centerOfMass[3];
	float m_totalMass;
/// [0,1]
	float m_stiffness;
	int m_numPoints;
	int m_numEdges;

public:

	ShapeMatchingRegion();
	
	void createRegion(const RegionVE& prof,
				const float* pos, const float* invMass);
	void setStiffness(const float& x);
				
	void updateRegion(const float* pos);

	const float* goalPosition(const int& i) const;
	
	const int& numPoints() const;
	const int& numEdges() const;
	const float& stiffness() const;
/// i-th edge
	void getEdge(int& v1, int& v2, const int& i) const;
	void getGoalInd(int& v1, int& v2, const int& i) const;
	
/// i-th q_n1 j-th goal position
/// x <- q_n1 + (g - q_n1) * k
	void solvePositionConstraint(float* x, const float* q_n1, 
			const int& i, const int& j) const;
	
protected:

/// x_cm <- sigma (m_i x_i) / sigma (mi)
/// p_i <- x_i - x_cm
	void updateCenterOfMass(const float* pos);
	
	void updateGoalPosition();
	
private:
	
	int findGoalInd(const int& x) const;
	
};

}

}
#endif
