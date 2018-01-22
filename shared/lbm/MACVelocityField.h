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
/// in grid
	int m_numParticles;
	
public:

	MACVelocityField();
/// (u, w) <- 0 in d-th dimension with block offset
	void resetBlockVelocity(float* u, float* sum, const int& iblock, const int& d);
	void depositeVelocities(int& countAdded, const float* vel, const float* pos, const int& np, 
				const float& scaling);
/// Jacobi relaxation iteration of six neighbors
/// u <- a u + b (ui-+1j-+1k-+1)
	void jacobi(const float& a, const float& b);
/// centered in other dimensions
	void getMarkerCoordWeight(int& i, int& j, int& k,
				float& barx, float& bary, float& barz,
				const float* u, const int& d) const;
/// u_ijk
	void evaluateCellCenterVelocity(float* u, 
				int& i, int& j, int& k) const;
/// U(p) relative to cell center
	void evaluateVelocityAtPosition(float* u, const float* p) const;
/// u is vec3 array
	void extractCellVelocities(float* u) const;
	const int& numParticlesInGrid() const;
/// M+1
/// N+1
/// P+1	
	static int MarkerDim[3];
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
	virtual void limitSpeed(float& x) const;
	
private:
	static const int ComponentMarkerTable[3][3];
	
	void addVelocity(float* u, float* sum,
			const int& i, const int& j, const int& k,
			const float& xbary, const float& ybary, const float& zbary,
			const float& q,
			const int& d);
/// u_ijk d-th component
	void evaluateCellCenterVelocityComponent(float& u, 
				const int& i, const int& j, const int& k,
				const int& d) const;
/// in z slice
	void sliceDepositeVelocities(int* count, const float* vel, const float* pos, const int& np, 
				const float& scaling, const int& zslice);
/// u <- u + w_i q_i 
/// w <- w + w_i
	void depositeVelocity(const float* pos, const float* vel, const float& scaling);
/// u <- u / w
	void finishDepositeVelocity();
	static void qdwei(float* q, float* wei, const int& ibegin, const int& iend );
	
};

}

}

#endif