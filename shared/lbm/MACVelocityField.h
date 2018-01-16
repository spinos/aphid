/*
 *  MACVelocityField.h
 *  
 *  Marker-And-Cell scheme for (u, v, w)
 *  dimension M x N x P
 *  http://www.thevisualroom.com/marker_and_cell_method.html
 *  http://rlguy.com/gridfluidsim/
 *
 *  Created by jian zhang on 1/17/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_MAC_VELOCITY_FIELD_H
#define APH_LBM_MAC_VELOCITY_FIELD_H

#include "UniformGrid.h"

namespace aphid {

namespace lbm {

class MACVelocityField : public UniformGrid {

/// marker velocity
	float* m_u[3];
/// marker sum of weight
	float* m_sum[3];
	
public:

	MACVelocityField();
/// (u, w) <- 0 in d-th dimension with block offset
	void resetBlockVelocity(float* u, float* sum, const int& iblock, const int& d);
/// u <- u + w_i q_i 
/// w <- w + w_i
	void depositeVelocity(const float* pos, const float* vel);
/// u <- u / w
	void finishDepositeVelocity();
/// centered in other dimensions
	void getMarkerCoordWeight(int& i, int& j, int& k,
				float& barx, float& bary, float& barz,
				const float* u, const int& d) const;
/// u_ijk
	void evaluateCellCenterVelocity(float* u, 
				int& i, int& j, int& k) const;
/// U(p) interpolate by centered coord
	void evaluateVelocityAtPosition(float* u, const float* p) const;
/// u is vec3 array
	void extractCellVelocities(float* u) const;
	
/// (M+1)NP
/// M(N+1)P
/// MN(P+1)
	static int BlockMarkerLength[3];

/// markers in d-th dimension
	static int MarkerInd(const int& i, const int& j, const int& k, const int& d);
				
protected:

/// all zero
	void clearVelocities();
	void depositeCellCenterVelocity(const int& i, const int& j, const int& k,
				const float* vel);
	
private:
	static const int ComponentMarkerTable[3][3];
	
	void addVelocity(float* u, float* sum,
			const int& i, const int& j, const int& k,
			const float& xbary, const float& ybary, const float& zbary,
			const float& q,
			const int& d);
	void qdwei(float* q, float* wei, const int& n );
/// d-th component
	void evaluateCellCenterVelocityComponent(float& u, 
				const int& i, const int& j, const int& k,
				const int& d) const;
	
};

}

}

#endif